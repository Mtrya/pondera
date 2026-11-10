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
    out, state = _linear_chunked_batched(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), chunk
    )
    return out[0], state[0]


def _linear_chunked_batched(q, k, v, chunk):
    """Independent document states along the leading batch dimension."""
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v = (x.float() for x in (q, k, v))
        group = q.shape[2] // k.shape[2]
        if group > 1:
            k = k.repeat_interleave(group, dim=2)
            v = v.repeat_interleave(group, dim=2)
        batch, length, heads, dim = q.shape
        width = v.shape[-1]
        pad = (-length) % chunk
        if pad:
            q = torch.cat([q, q.new_zeros(batch, pad, heads, dim)], dim=1)
            k = torch.cat([k, k.new_zeros(batch, pad, heads, dim)], dim=1)
            v = torch.cat([v, v.new_zeros(batch, pad, heads, width)], dim=1)
        count = (length + pad) // chunk
        q = q.reshape(batch, count, chunk, heads, dim).transpose(2, 3)
        k = k.reshape(batch, count, chunk, heads, dim).transpose(2, 3)
        v = v.reshape(batch, count, chunk, heads, width).transpose(2, 3)
        scores = q @ k.mT
        mask = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device))
        scores = scores.masked_fill(~mask, 0.0)
        intra = scores @ v
        states = k.mT @ v
        exclusive = torch.cat(
            [torch.zeros_like(states[:, :1]), torch.cumsum(states[:, :-1], dim=1)],
            dim=1,
        )
        inter = q @ exclusive
        out = (
            (intra + inter).transpose(2, 3).reshape(batch, count * chunk, heads, width)
        )
        return out[:, :length], states.sum(dim=1)


def _prepare_linear(cu_seqlens, device, chunk=CHUNK):
    """Group documents by powers of two chunks, with less than 2× chunk padding."""
    cu = torch.as_tensor(cu_seqlens, dtype=torch.int64, device="cpu").tolist()
    if len(cu) == 2:
        return None
    groups = {}
    for start, stop in zip(cu[:-1], cu[1:]):
        count = (stop - start + chunk - 1) // chunk
        length = chunk * (1 << (count - 1).bit_length())
        groups.setdefault(length, []).append((start, stop - start))
    buckets, destinations = [], []
    for length, documents in groups.items():
        starts, lengths = torch.tensor(documents).unbind(dim=1)
        offsets = torch.arange(length)
        valid = offsets.unsqueeze(0) < lengths.unsqueeze(1)
        indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
        selected = valid.flatten().nonzero().flatten()
        destinations.append(indices.flatten()[selected])
        buckets.append(
            tuple(
                tensor.to(device, non_blocking=True)
                for tensor in (indices.masked_fill(~valid, 0), valid, selected)
            )
        )
    restore = torch.empty(cu[-1], dtype=torch.int64)
    if destinations:
        restore[torch.cat(destinations)] = torch.arange(cu[-1])
    return buckets, restore.to(device, non_blocking=True)


def _linear_varlen(q, k, v, prepared, chunk=CHUNK):
    if prepared is None:
        return linear_chunked(q, k, v, chunk)[0].to(q.dtype)
    buckets, restore = prepared
    if not buckets:
        return q.new_empty(0, *q.shape[1:])
    outputs = []
    for indices, valid, selected in buckets:
        qs, ks, vs = (
            x[indices].masked_fill(~valid[..., None, None], 0) for x in (q, k, v)
        )
        out, _ = _linear_chunked_batched(qs, ks, vs, chunk)
        outputs.append(out.flatten(0, 1)[selected])
    return torch.cat(outputs)[restore].to(q.dtype)


def linear_varlen_attention(q, k, v, cu_seqlens, chunk=CHUNK):
    return _linear_varlen(q, k, v, _prepare_linear(cu_seqlens, q.device, chunk), chunk)


class LinearSelfAttention(CausalSelfAttention):
    """Same projections and shapes as the softmax core; bare linear kernel."""

    prepare = staticmethod(_prepare_linear)

    def forward(self, x, prepared):
        q, k, v = self._project(x)
        return self.o(_linear_varlen(q, k, v, prepared).reshape(x.shape))

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
    architecture = "linear"
