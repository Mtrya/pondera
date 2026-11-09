"""Bare linear attention: identity feature map, unnormalized, no decay.

out_t = sum_{i<=t} (q_t · k_i) v_i. The state S = sum k_i ⊗ v_i is always
fp32 — without decay it grows with sequence length, and a bf16 snapshot
would quantize away later updates. Training uses the chunked parallel
form; generation uses the constant-size recurrent state; both consume the
same cu_seqlens with state zeroed at document boundaries.
"""

import torch

from pondera.models.vanilla import CausalSelfAttention, VanillaModel

CHUNK = 64


def linear_recurrent(q, k, v):
    """Per-token recurrence, the exact reference: out_t = q_t @ S_t."""
    q, k, v = (x.float() for x in (q, k, v))
    state = q.new_zeros(q.shape[1], q.shape[2], v.shape[2])
    outputs = []
    for index in range(q.shape[0]):
        state = state + k[index].unsqueeze(-1) * v[index].unsqueeze(-2)
        outputs.append((q[index].unsqueeze(-2) @ state).squeeze(-2))
    return torch.stack(outputs), state


def linear_chunked(q, k, v, chunk=CHUNK):
    """Chunked parallel form of the same recurrence, fp32 throughout."""
    q, k, v = (x.float() for x in (q, k, v))
    group = q.shape[1] // k.shape[1]
    if group > 1:
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
    length, heads, dim = q.shape
    width = v.shape[-1]
    pad = (-length) % chunk
    if pad:
        q = torch.cat([q, q.new_zeros(pad, heads, dim)])
        k = torch.cat([k, k.new_zeros(pad, heads, dim)])
        v = torch.cat([v, v.new_zeros(pad, heads, width)])
    count = (length + pad) // chunk
    q = q.view(count, chunk, heads, dim).transpose(1, 2)
    k = k.view(count, chunk, heads, dim).transpose(1, 2)
    v = v.view(count, chunk, heads, width).transpose(1, 2)
    scores = q @ k.mT
    mask = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device))
    scores = scores.masked_fill(~mask, 0.0)
    intra = scores @ v
    states = k.mT @ v
    exclusive = torch.cumsum(states, dim=0) - states
    inter = q @ exclusive
    out = (intra + inter).transpose(1, 2).reshape(count * chunk, heads, width)
    return out[:length], states.sum(dim=0)


def linear_varlen_attention(q, k, v, cu_seqlens, chunk=CHUNK):
    outputs = []
    for start, stop in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        out, _ = linear_chunked(q[start:stop], k[start:stop], v[start:stop], chunk)
        outputs.append(out)
    if not outputs:
        return q.new_empty(0, *q.shape[1:])
    return torch.cat(outputs).to(q.dtype)


class LinearSelfAttention(CausalSelfAttention):
    """Same projections and shapes as the softmax core; bare linear kernel."""

    def forward(self, x, cu_seqlens):
        q, k, v = self._project(x)
        return self.o(linear_varlen_attention(q, k, v, cu_seqlens).reshape(x.shape))

    def forward_cached(self, x, cu_seqlens, cache=None):
        q, k, v = self._project(x)
        outputs, states = [], []
        for start, stop in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            qs, ks, vs = q[start:stop], k[start:stop], v[start:stop]
            if cache is not None and qs.shape[0] == 1:
                group = qs.shape[1] // ks.shape[1]
                ks = ks.repeat_interleave(group, dim=1)
                vs = vs.repeat_interleave(group, dim=1)
                state = cache + ks[0].float().unsqueeze(-1) * vs[0].float().unsqueeze(
                    -2
                )
                out = (qs[0].float().unsqueeze(-2) @ state).squeeze(-2).unsqueeze(0)
            else:
                out, state = linear_chunked(qs, ks, vs)
            outputs.append(out)
            states.append(state)
        return self.o(torch.cat(outputs).to(x.dtype).reshape(x.shape)), states[0]


class LinearVanillaModel(VanillaModel):
    attention_cls = LinearSelfAttention
