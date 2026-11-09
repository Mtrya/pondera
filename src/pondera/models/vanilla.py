"""Vanilla causal transformer over one packed token stream.

Full causal attention, NoPE, pre-norm RMSNorm, SiTU-GLU feed-forward, and
grouped-query attention. No position encoding of any kind; the causal mask
is the only positional signal. The forward path consumes packed documents
through ``cu_seqlens`` (the Phase 1 batch contract) and never materializes
an L×L mask.
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
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


def varlen_attention(q, k, v, cu_seqlens, is_causal=True):
    """Causal GQA over packed documents: [T, H, D] × [T, H_kv, D]² → [T, H, D].

    CUDA half-precision uses the fused varlen flash kernel; everything else
    falls back to the per-document eager path (also the fp32 reference).
    Incremental decode (one query against a longer history) is always eager:
    a single query already sees its whole past, so no causal mask applies.
    """
    if is_causal and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        max_q = max_k = int(
            (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            if len(cu_seqlens) > 1
            else 0
        )
        return torch.ops.aten._flash_attention_forward(
            q, k, v, cu_seqlens, cu_seqlens, max_q, max_k, 0.0, True, False
        )[0]
    return eager_varlen_attention(q, k, v, cu_seqlens, is_causal)


class CausalSelfAttention(nn.Module):
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

    def _project(self, x):
        shape = (x.shape[0], self.num_kv_heads, self.head_dim)
        return (
            self.q(x).view(x.shape[0], self.num_heads, self.head_dim),
            self.k(x).view(shape),
            self.v(x).view(shape),
        )

    def forward(self, x, cu_seqlens):
        q, k, v = self._project(x)
        return self.o(varlen_attention(q, k, v, cu_seqlens).reshape(x.shape))

    def forward_cached(self, x, cu_seqlens, cache=None):
        """One full sequence or an incremental step; cache is per-layer (k, v)."""
        q, k, v = self._project(x)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=0)
            v = torch.cat([cache[1], v], dim=0)
        out = self.o(
            varlen_attention(q, k, v, cu_seqlens, q.shape[0] == k.shape[0]).reshape(
                x.shape
            )
        )
        return out, (k, v)


class Block(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, num_kv_heads):
        super().__init__()
        self.norm_attention = RMSNorm(hidden_size)
        self.attention = CausalSelfAttention(hidden_size, num_heads, num_kv_heads)
        self.norm_ffn = RMSNorm(hidden_size)
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def _ffn(self, x):
        return self.down(situ_glu(self.gate(x), self.up(x)))

    def forward(self, x, cu_seqlens):
        x = x + self.attention(self.norm_attention(x), cu_seqlens)
        return x + self._ffn(self.norm_ffn(x))

    def forward_cached(self, x, cu_seqlens, cache=None):
        attended, new_cache = self.attention.forward_cached(
            self.norm_attention(x), cu_seqlens, cache
        )
        x = x + attended
        return x + self._ffn(self.norm_ffn(x)), new_cache


class VanillaModel(nn.Module):
    """The frozen vanilla definition; only the attention core varies in Phase 3."""

    def __init__(
        self,
        num_blocks,
        hidden_size,
        intermediate_size,
        num_heads,
        num_kv_heads,
        vocab_size,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        self.blocks = nn.ModuleList(
            Block(hidden_size, intermediate_size, num_heads, num_kv_heads)
            for _ in range(num_blocks)
        )
        self.norm = RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, cu_seqlens):
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x, cu_seqlens)
        return self.head(self.norm(x))

    def forward_cached(self, input_ids, cu_seqlens, caches=None):
        x = self.token_embedding(input_ids)
        new_caches = []
        for index, block in enumerate(self.blocks):
            cache = None if caches is None else caches[index]
            x, new_cache = block.forward_cached(x, cu_seqlens, cache)
            new_caches.append(new_cache)
        return self.head(self.norm(x)), new_caches

    def non_embedding_parameters(self):
        return (
            sum(p.numel() for p in self.parameters())
            - self.token_embedding.weight.numel()
        )
