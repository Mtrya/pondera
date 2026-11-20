"""Gated DeltaNet-2 with independent channel-wise erase and write gates.

The erase projection follows the author's beta_guard, including its
straight-through gradient. Training and decoding both use the per-token
log decay directly, avoiding reconstruction from chunk cumulative sums.
"""

import torch
from torch import nn
from torch.nn import functional as F

from pondera.models.delta import GatedDeltaModel, KimiDeltaAttention


def guard_erase(key, erase, log_decay):
    """Contract erase contrast; identity erase gradient, no projection k/g gradient."""
    with torch.no_grad():
        k2 = key.float().square()
        b = erase.float()
        n = k2.sum(-1, keepdim=True)
        a = (b * k2).sum(-1, keepdim=True)
        nu = (b.square() * k2).sum(-1, keepdim=True)
        contrast = (n * nu - a.square()).clamp_min(0)
        c2 = (-2 * log_decay.float().amax(-1, keepdim=True)).exp()

        def threshold(moment):
            return ((c2 - 1) * (1 - (1 - moment).square() / c2)).clamp_min(0)

        critical = threshold(a)
        unsafe = (n > 1e-20) & (contrast > critical)
        mean = a / n.clamp_min(1e-20)
        factor = (
            ((1 - 1 / 32) * critical / contrast.clamp_min(1e-30)).sqrt().clamp(0, 1)
        )
        candidate = (
            torch.where(unsafe, mean + factor * (b - mean), b).to(erase.dtype).float()
        )
        aq = (candidate * k2).sum(-1, keepdim=True)
        nuq = (candidate.square() * k2).sum(-1, keepdim=True)
        contrastq = (n * nuq - aq.square()).clamp_min(0)
        tolerance = 4 * torch.finfo(erase.dtype).eps * (n * nuq + aq.square())
        fallback = unsafe & (contrastq > threshold(aq) + tolerance)
        projected = torch.where(fallback, aq / n.clamp_min(1e-20), candidate).to(
            erase.dtype
        )
    # Add exact zero so straight-through differentiation preserves BF16 bits.
    return projected + (erase - erase.detach())


def gdn2_attention(
    q, k, v, g, erase, write, prepared, cache=None, output_final_state=False
):
    """Packed GDN-2 memory; kernels consume its diagonal-plus-rank-one update."""
    q, k = (
        (x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)).to(
            x.dtype
        )
        for x in (q, k)
    )
    erase = guard_erase(k, erase.to(k.dtype), g)
    a = (-erase.float() * k.float() * g.exp()).to(k.dtype)
    value = (write.to(v.dtype).float() * v.float()).to(v.dtype)
    cu_cpu, cu_device = prepared
    if q.device.type in ("cuda", "npu"):
        from fla.ops.generalized_delta_rule.dplr import (
            chunk_dplr_delta_rule,
            fused_recurrent_dplr_delta_rule,
        )

        incremental = not torch.is_grad_enabled() and (
            cache is not None or q.dtype == torch.float32
        )
        kernel = (
            fused_recurrent_dplr_delta_rule if incremental else chunk_dplr_delta_rule
        )
        options = {} if incremental else {"cu_seqlens_cpu": cu_cpu}
        output, state = kernel(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=value.unsqueeze(0),
            a=a.unsqueeze(0),
            b=k.unsqueeze(0),
            gk=g.unsqueeze(0),
            initial_state=cache,
            output_final_state=output_final_state,
            cu_seqlens=cu_device,
            **options,
        )
        return output.squeeze(0), state

    q, k, value, a, g = (x.float() for x in (q, k, value, a, g))
    outputs, states = [], []
    scale = q.shape[-1] ** -0.5
    for index, (start, stop) in enumerate(
        zip(cu_cpu[:-1].tolist(), cu_cpu[1:].tolist())
    ):
        state = (
            q.new_zeros(q.shape[1], q.shape[2], v.shape[2])
            if cache is None
            else cache[index].float()
        )
        for token in range(start, stop):
            correction = (state * a[token].unsqueeze(-1)).sum(-2)
            state = state * g[token].exp().unsqueeze(-1) + k[token].unsqueeze(-1) * (
                value[token] + correction
            ).unsqueeze(-2)
            outputs.append((q[token].unsqueeze(-1) * state).sum(-2) * scale)
        states.append(state)
    return torch.stack(outputs).to(v.dtype), torch.stack(
        states
    ) if output_final_state else None


class GatedDeltaNet2Attention(KimiDeltaAttention):
    channel_erase = True

    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__(hidden_size, num_heads, num_kv_heads)
        self.write = nn.Linear(hidden_size, hidden_size, bias=False)

    def _memory(self, vectors, x, prepared, cache=None, output_final_state=False):
        q, k, v = (tensor.view(-1, self.num_heads, self.head_dim) for tensor in vectors)
        g, erase = self._gates(x)
        write = self.write(x).float().sigmoid().view_as(v)
        return gdn2_attention(
            q, k, v, g, erase.view_as(k), write, prepared, cache, output_final_state
        )

    def _output(self, x, memory):
        gate = F.silu(self.gate(x).view_as(memory))
        return self.o((self.norm_output(memory) * gate).reshape_as(x))


class GatedDeltaNet2Model(GatedDeltaModel):
    architecture = "gdn2"
    attention_cls = GatedDeltaNet2Attention
