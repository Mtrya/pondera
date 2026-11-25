"""Packed causal grouped-query softmax attention and block operations."""

from functools import cache

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        if x.device.type == "npu" and x.dtype == self.weight.dtype == torch.float32:
            import torch_npu

            return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight


def situ_glu(gate, up):
    """SiTU-GLU: (4·tanh(g/4) ⊙ σ(g)) ⊙ (25·tanh(u/25)), |output| ≤ 100."""
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
        25.0 * torch.tanh(up / 25.0)
    )


def eager_varlen_attention(q, k, v, cu_seqlens, is_causal=True):
    """Per-document causal SDPA; the exact reference for the fused path."""
    outputs = []
    group = q.shape[1] // k.shape[1]
    for start, stop in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        qs = q[start:stop].transpose(0, 1).unsqueeze(0)
        ks = k[start:stop].repeat_interleave(group, dim=1).transpose(0, 1).unsqueeze(0)
        vs = v[start:stop].repeat_interleave(group, dim=1).transpose(0, 1).unsqueeze(0)
        outputs.append(
            nn.functional.scaled_dot_product_attention(qs, ks, vs, is_causal=is_causal)
            .squeeze(0)
            .transpose(0, 1)
        )
    return torch.cat(outputs) if outputs else q.new_empty(0, *q.shape[1:])


@cache
def _npu_causal_mask(device):
    # Reuse across layers, including when inference precedes training.
    with torch.inference_mode(False):
        return torch.ones((2048, 2048), dtype=torch.bool, device=device).triu_(1)


def _npu_varlen_attention(q, k, v, ends):
    import torch_npu

    if not ends:
        return q.new_empty(0, *q.shape[1:])
    mask = _npu_causal_mask(q.device)
    outputs = []
    start = 0
    # Bound each operator call's document count without changing the batch.
    for offset in range(0, len(ends), 1024):
        group = ends[offset : offset + 1024]
        stop = group[-1]
        local_ends = [end - start for end in group] if start else group
        outputs.append(
            torch_npu.npu_fusion_attention(
                q[start:stop],
                k[start:stop],
                v[start:stop],
                head_num=q.shape[1],
                input_layout="TND",
                scale=q.shape[-1] ** -0.5,
                keep_prob=1.0,
                actual_seq_qlen=local_ends,
                actual_seq_kvlen=local_ends,
                atten_mask=mask,
                sparse_mode=2,
            )[0]
        )
        start = stop
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


def varlen_attention(
    q, k, v, cu_seqlens, is_causal=True, max_seqlen=None, sequence_ends=None
):
    """Causal GQA over packed documents: [T, H, D] × [T, H_kv, D]² → [T, H, D].

    CUDA and NPU half-precision use fused varlen kernels; everything else
    falls back to the per-document eager path (also the fp32 reference).
    Incremental decode (one query against a longer history) is always eager:
    a single query already sees its whole past, so no causal mask applies.
    """
    if is_causal and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        if max_seqlen is None:
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
        return torch.ops.aten._flash_attention_forward(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, 0.0, True, False
        )[0]
    if (
        is_causal
        and q.device.type == "npu"
        and q.dtype in (torch.float16, torch.bfloat16)
    ):
        if sequence_ends is None:
            sequence_ends = cu_seqlens[1:].tolist()
        return _npu_varlen_attention(q, k, v, sequence_ends)
    return eager_varlen_attention(q, k, v, cu_seqlens, is_causal)


class CausalSelfAttention(nn.Module):
    @staticmethod
    def prepare(cu_seqlens, device):
        """Compute shared sequence metadata on the host once per forward."""
        cu = torch.as_tensor(cu_seqlens, dtype=torch.int32, device="cpu")
        lengths = cu[1:] - cu[:-1]
        maximum = int(lengths.max()) if lengths.numel() else 0
        device = torch.device(device)
        ends = cu[1:].tolist() if device.type == "npu" else None
        if device.type == "cuda":
            cu = cu.to(device, non_blocking=True)
        return cu, maximum, ends

    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()
        if hidden_size % num_heads or num_heads % num_kv_heads:
            raise ValueError("Head counts must divide each other and the width")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)

    def _project(self, x, *, value_input=None):
        shape = (x.shape[0], self.num_kv_heads, self.head_dim)
        return (
            self.q(x).view(x.shape[0], self.num_heads, self.head_dim),
            self.k(x).view(shape),
            self.v(x if value_input is None else value_input).view(shape),
        )

    def forward(self, x, prepared):
        q, k, v = self._project(x)
        return self.attend_projected(q, k, v, prepared)

    def attend_projected(self, q, k, v, prepared):
        cu_seqlens, maximum, ends = prepared
        return self.o(
            varlen_attention(
                q, k, v, cu_seqlens, max_seqlen=maximum, sequence_ends=ends
            ).flatten(1)
        )

    def forward_cached(self, x, cu_seqlens, cache=None):
        """One full sequence or an incremental step; cache is per-layer (k, v)."""
        q, k, v = self._project(x)
        return self.attend_projected_cached(q, k, v, cu_seqlens, cache)

    def attend_projected_cached(self, q, k, v, cu_seqlens, cache):
        if cache is not None:
            k = torch.cat([cache[0], k], dim=0)
            v = torch.cat([cache[1], v], dim=0)
        out = self.o(
            varlen_attention(q, k, v, cu_seqlens, q.shape[0] == k.shape[0]).flatten(1)
        )
        return out, (k, v)
