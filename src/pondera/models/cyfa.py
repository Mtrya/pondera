"""Cyclic Flow Attention through two scalar-decay memory recurrences.

The Fourier formulation follows https://arxiv.org/abs/2609.36259. The key
memory reads relative-time slot scores; a softmax readout then queries the
separate value memory. Queries and keys are RMS-normalized by the caller.
"""

import math
from functools import cache, partial

import torch
from torch import nn
from torch.nn import functional as F

from pondera.models.memory import HybridModel, RecurrentAttention
from pondera.models.vanilla import RMSNorm


class _Clock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, delta, prepared, initial, period):
        # Sigmoid increments lie in [0, 1]. 44 fractional bits represent every
        # fp32 value >= 2**-21 exactly; rounding smaller values introduces at
        # most 2**-45 per token. Integer addition makes cached and parallel
        # sums identical. Bounded chunks and periodic reduction avoid overflow.
        unit = 1 << 44
        cpu_boundaries, device_boundaries = prepared
        pieces, states = [], []
        for doc, (start, end) in enumerate(
            zip(cpu_boundaries[:-1].tolist(), cpu_boundaries[1:].tolist())
        ):
            carry = 0 if initial is None else initial[doc]
            for offset in range(start, end, 65536):
                increments = (
                    (delta[offset : min(offset + 65536, end)] * unit).round().long()
                )
                ticks = (increments.cumsum(0) + carry).remainder(period * unit)
                pieces.append(ticks.float() / unit)
                carry = ticks[-1]
            states.append(carry)
        ctx.save_for_backward(device_boundaries)
        ctx.cpu_boundaries = cpu_boundaries
        return torch.cat(pieces), torch.stack(states)

    @staticmethod
    def backward(ctx, gradient, _state_gradient):
        (boundaries,) = ctx.saved_tensors
        if gradient.device.type in {"cuda", "npu"}:
            from fla.ops.utils.cumsum import chunk_global_cumsum

            result = chunk_global_cumsum(
                gradient.contiguous().unsqueeze(0),
                reverse=True,
                cu_seqlens=boundaries,
            ).squeeze(0)
        else:
            cu = ctx.cpu_boundaries.tolist()
            result = torch.cat(
                [
                    gradient[a:b].flip(0).cumsum(0).flip(0)
                    for a, b in zip(cu[:-1], cu[1:])
                ]
            )
        return result, None, None, None


@cache
@torch.inference_mode(False)
def _basis(slots, device):
    """Real Fourier basis, with the zero-frequency sine padding column."""
    period = slots - 1
    frequency = torch.arange(slots // 2, device=device, dtype=torch.float32)
    position = torch.arange(period, device=device, dtype=torch.float32)
    phase = position[:, None] * frequency * (2 * math.pi / period)
    scale = torch.full_like(frequency, (2 / period) ** 0.5)
    scale[0] = period**-0.5
    return (torch.stack((phase.cos(), phase.sin()), -1) * scale[None, :, None]).flatten(
        1
    )


def _rotation(clock, slots):
    period = slots - 1
    frequency = torch.arange(slots // 2, device=clock.device)
    integer = clock.floor()
    fraction = clock - integer
    # Reducing integer turns before multiplication preserves fractional phase
    # on long sequences. The differentiable fraction retains the clock gradient.
    turns = (integer.long().remainder(period)[..., None] * frequency).remainder(period)
    turns = turns.float() + fraction[..., None] * frequency
    turns = torch.where(turns > period / 2, turns - period, turns)
    angle = turns * (2 * math.pi / period)
    return angle.cos(), angle.sin()


def _rotate(x, cosine, sine):
    pairs = x.unflatten(-1, (-1, 2))
    real, imaginary = pairs[..., 0], pairs[..., 1]
    return torch.stack(
        (real * cosine - imaginary * sine, real * sine + imaginary * cosine), -1
    ).flatten(-2)


def _scalar_memory(q, k, v, g, scale, prepared, initial, output_final):
    cpu_boundaries, device_boundaries = prepared
    if (
        q.device.type == "npu"
        and int((cpu_boundaries[1:] - cpu_boundaries[:-1]).max()) > 4096
    ):
        # Ascend's GLA state kernel statically unrolls each document's chunks.
        # Bound compilation while carrying both state and its gradient across
        # segments. This preserves the complete document's causal context.
        pieces, states = [], []
        for doc, (start, end) in enumerate(
            zip(cpu_boundaries[:-1].tolist(), cpu_boundaries[1:].tolist())
        ):
            state = None if initial is None else initial[doc : doc + 1]
            for offset in range(start, end, 4096):
                stop = min(offset + 4096, end)
                bounds = RecurrentAttention.prepare([0, stop - offset], q.device)
                part, state = _scalar_memory(
                    q[offset:stop],
                    k[offset:stop],
                    v[offset:stop],
                    g[offset:stop],
                    scale,
                    bounds,
                    state,
                    True,
                )
                pieces.append(part)
            states.append(state)
        return torch.cat(pieces), torch.cat(states) if output_final else None
    if q.device.type in {"cuda", "npu"}:
        from fla.ops.simple_gla import (
            chunk_simple_gla,
            fused_recurrent_simple_gla,
        )

        incremental = q.shape[0] == 1 or (
            not torch.is_grad_enabled() and q.dtype == torch.float32
        )
        function = fused_recurrent_simple_gla if incremental else chunk_simple_gla
        output, state = function(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            g=g.unsqueeze(0),
            scale=scale,
            initial_state=initial,
            output_final_state=output_final,
            cu_seqlens=device_boundaries,
        )
        return output.squeeze(0), state
    outputs, states = [], []
    for document, (start, end) in enumerate(
        zip(cpu_boundaries[:-1], cpu_boundaries[1:])
    ):
        state = (
            q.new_zeros(q.shape[1], q.shape[2], v.shape[2], dtype=torch.float32)
            if initial is None
            else initial[document].float()
        )
        for token in range(start, end):
            state = state * g[token].float().exp()[:, None, None]
            state = state + k[token].float()[:, :, None] * v[token].float()[:, None, :]
            outputs.append((q[token].float()[:, :, None] * state).sum(-2) * scale)
        states.append(state)
    return torch.stack(outputs).to(v.dtype), torch.stack(
        states
    ) if output_final else None


def cyclic_flow_attention(
    q,
    k,
    v,
    g,
    delta,
    beta,
    readout,
    prepared,
    initial=None,
    output_final=False,
):
    """Packed [T,H,D] normalized Q/K/V, [T,H] gates, [H,S-1,S-1] readout.

    State is (key memory [N,H,D,S], value memory [N,H,S,D], int64 clock [N,H]).
    The even storage width S includes one unused Fourier padding column.
    """
    slots = readout.shape[-1] + 1
    if slots % 2 or slots < 2:
        raise ValueError("CyFA storage width must be positive and even")
    key_state, value_state, clock_state = (
        (None, None, None) if initial is None else initial
    )
    with torch.autocast(q.device.type, enabled=False):
        clock, clock_state = _Clock.apply(
            delta.float(), prepared, clock_state, slots - 1
        )
        cosine, sine = _rotation(clock, slots)
        scale = torch.full((slots // 2,), (2 / (slots - 1)) ** 0.5, device=q.device)
        scale[0] = (slots - 1) ** -0.5
        atom = torch.stack((cosine, -sine), -1).flatten(-2)
        write = (atom * scale.repeat_interleave(2) * beta[..., None]).to(q.dtype)
        table = readout.float() @ _basis(slots, q.device)
        logits, key_state = _scalar_memory(
            q,
            k,
            write,
            g,
            q.shape[-1] ** -0.5,
            prepared,
            key_state,
            output_final,
        )
        relative_logits = _rotate(logits.float(), cosine, sine)
        probabilities = torch.einsum("ths,hrs->thr", relative_logits, table).softmax(-1)
        coefficients = torch.einsum("thr,hrs->ths", probabilities, table)
        query = _rotate(coefficients, cosine, -sine).to(v.dtype)
        output, value_state = _scalar_memory(
            query,
            write,
            v,
            g,
            1.0,
            prepared,
            value_state,
            output_final,
        )
        state = (key_state, value_state, clock_state) if output_final else None
    return output, state


class CyclicFlowAttention(RecurrentAttention):
    """Relative-time key/value memories with the original CyFA frontend."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, *, num_slots):
        super().__init__(hidden_size, num_heads, activation=None)
        if num_slots < 2 or num_slots % 2:
            raise ValueError("CyFA storage width must be positive and even")
        self.num_slots = num_slots
        self.decay = nn.Linear(hidden_size, num_heads, bias=False)
        dt = torch.exp(torch.empty(num_heads).uniform_(math.log(0.001), math.log(0.1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.clock = nn.Linear(hidden_size, num_heads, bias=True)
        self.beta = nn.Linear(hidden_size, num_heads, bias=True)
        self.A_log = nn.Parameter(torch.empty(num_heads).uniform_(0, 16).log())
        self.readout = nn.Parameter(
            torch.eye(num_slots - 1).expand(num_heads, -1, -1).clone()
        )
        self.q_norm = RMSNorm(self.head_dim, eps=1e-5)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-5)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, self.head_dim, bias=False),
            nn.Linear(self.head_dim, hidden_size, bias=True),
        )
        self.norm_output = RMSNorm(self.head_dim, eps=1e-5)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)

    def _memory(self, vectors, x, prepared, cache=None, output_final_state=False):
        q, k, v = (vector.view(-1, self.num_heads, self.head_dim) for vector in vectors)
        q, k = self.q_norm(q.float()).to(q.dtype), self.k_norm(k.float()).to(k.dtype)
        decay = -self.A_log.exp() * F.softplus(self.decay(x).float() + self.dt_bias)
        # Stop gradient into the residual stream, while learning clock weights.
        delta = self.clock(x.detach()).float().sigmoid()
        beta = self.beta(x).float().sigmoid()
        return cyclic_flow_attention(
            q,
            k,
            v,
            decay,
            delta,
            beta,
            self.readout,
            prepared,
            cache,
            output_final_state,
        )

    def _output(self, x, memory):
        # Slot averaging shrinks activations; restore scale before RMS epsilon.
        normalized = self.norm_output(memory.float() * (self.num_slots - 1))
        gate = self.gate(x).float().sigmoid().view_as(memory)
        return self.o((normalized * gate).to(memory.dtype).reshape_as(x))


class CyclicFlowModel(HybridModel):
    architecture = "cyfa"

    def __init__(self, *, num_slots=64, **kwargs):
        self.attention_cls = partial(CyclicFlowAttention, num_slots=num_slots)
        super().__init__(**kwargs)
