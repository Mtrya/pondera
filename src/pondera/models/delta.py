"""Gated DeltaNet and Kimi Delta Attention over intact packed documents.

GDN decays the associative memory per head; KDA decays each key channel.
Both correct the decayed memory with the current key/value residual. Short
causal convolutions, Q/K L2 normalization and gated head normalization follow
the original architectures. Accelerator training uses FLA's chunk kernels;
the fp32 recurrence provides CPU execution and an independent reference.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from pondera.models.memory import HybridModel, RecurrentAttention
from pondera.models.vanilla import RMSNorm


def delta_recurrent(q, k, v, g, beta, initial_state=None):
    """One document, [T,H,D] vectors and fp32 [H,D,D] memory."""
    q, k, v, g, beta = (tensor.float() for tensor in (q, k, v, g, beta))
    state = (
        q.new_zeros(q.shape[1], q.shape[2], v.shape[2])
        if initial_state is None
        else initial_state.float()
    )
    if g.ndim == 2:
        g = g.unsqueeze(-1)
    outputs = []
    scale = q.shape[-1] ** -0.5
    for index in range(q.shape[0]):
        state = state * g[index].exp().unsqueeze(-1)
        residual = v[index] - (k[index].unsqueeze(-1) * state).sum(-2)
        state = state + (beta[index, :, None] * k[index]).unsqueeze(
            -1
        ) * residual.unsqueeze(-2)
        outputs.append((q[index].unsqueeze(-1) * state).sum(-2) * scale)
    return torch.stack(outputs), state


def delta_attention(
    q,
    k,
    v,
    g,
    beta,
    prepared,
    cache=None,
    output_final_state=False,
    *,
    channel_decay=False,
    normalize=True,
):
    """Packed delta-rule memory with scalar or key-channel decay."""
    cu_cpu, cu_device = prepared
    if q.device.type in ("cuda", "npu"):
        if channel_decay:
            from fla.ops.kda import chunk_kda, fused_recurrent_kda

            chunk, recurrent = chunk_kda, fused_recurrent_kda
        else:
            from fla.ops.gated_delta_rule import (
                chunk_gated_delta_rule,
                fused_recurrent_gated_delta_rule,
            )

            chunk, recurrent = (
                chunk_gated_delta_rule,
                fused_recurrent_gated_delta_rule,
            )
        incremental = not torch.is_grad_enabled() and (
            cache is not None or q.dtype == torch.float32
        )
        options = {} if incremental else {"cu_seqlens_cpu": cu_cpu}
        output, state = (recurrent if incremental else chunk)(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=cache,
            output_final_state=output_final_state,
            cu_seqlens=cu_device,
            use_qk_l2norm_in_kernel=normalize,
            **options,
        )
        return output.squeeze(0), state
    if normalize:
        q, k = (
            (
                tensor.float()
                * torch.rsqrt(tensor.float().square().sum(-1, keepdim=True) + 1e-6)
            ).to(tensor.dtype)
            for tensor in (q, k)
        )
    outputs, states = [], []
    for index, (start, stop) in enumerate(
        zip(cu_cpu[:-1].tolist(), cu_cpu[1:].tolist())
    ):
        output, state = delta_recurrent(
            q[start:stop],
            k[start:stop],
            v[start:stop],
            g[start:stop],
            beta[start:stop],
            None if cache is None else cache[index],
        )
        outputs.append(output)
        states.append(state)
    return torch.cat(outputs).to(v.dtype), torch.stack(
        states
    ) if output_final_state else None


def replay_delta_attention(q, k, v, g, beta, previous, prepared, **options):
    """Write this token's current and earlier-loop K/V, newest loop first.

    Decay is applied once, before the first write. Each write retains its
    originating beta; the current query reads after the oldest loop's write.
    Expanding each token into consecutive writes reuses the packed delta kernel.
    """
    if not previous:
        return delta_attention(q, k, v, g, beta, prepared, **options)
    writes = ((k, v, beta), *reversed(previous))
    count = len(writes)
    keys, values, strengths = (
        torch.stack(tensors, dim=1).flatten(0, 1) for tensors in zip(*writes)
    )
    queries = q.unsqueeze(1).expand(-1, count, -1, -1).flatten(0, 1)
    decay = torch.stack((g, *(torch.zeros_like(g) for _ in previous)), dim=1)
    output, state = delta_attention(
        queries,
        keys,
        values,
        decay.flatten(0, 1),
        strengths,
        tuple(cu * count for cu in prepared),
        **options,
    )
    return output[count - 1 :: count], state


class GatedDeltaAttention(RecurrentAttention):
    channel_decay = False
    channel_erase = False

    def __init__(self, hidden_size, num_heads, num_kv_heads):
        # Recurrent layers use full memory heads; num_kv_heads applies to softmax.
        super().__init__(hidden_size, num_heads)
        self.beta = nn.Linear(
            hidden_size, hidden_size if self.channel_erase else num_heads, bias=False
        )
        if self.channel_decay:
            self.decay = nn.Sequential(
                nn.Linear(hidden_size, self.head_dim, bias=False),
                nn.Linear(self.head_dim, hidden_size, bias=False),
            )
            self.gate = nn.Sequential(
                nn.Linear(hidden_size, self.head_dim, bias=False),
                nn.Linear(self.head_dim, hidden_size, bias=True),
            )
            gate_width = hidden_size
            minimum = 1.0
        else:
            self.decay = nn.Linear(hidden_size, num_heads, bias=False)
            self.gate = nn.Linear(hidden_size, hidden_size, bias=False)
            gate_width = num_heads
            minimum = 0.0
        self.A_log = nn.Parameter(torch.empty(num_heads).uniform_(minimum, 16).log())
        dt = torch.exp(torch.empty(gate_width).uniform_(math.log(0.001), math.log(0.1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.norm_output = RMSNorm(self.head_dim, eps=1e-5)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)

    def _gates(self, x):
        decay = F.softplus(self.decay(x).float() + self.dt_bias)
        if self.channel_decay:
            decay = decay.view(-1, self.num_heads, self.head_dim)
            decay = -self.A_log.exp()[None, :, None] * decay
        else:
            decay = -self.A_log.exp() * decay
        return decay, self.beta(x).float().sigmoid()

    def _memory(self, vectors, x, prepared, cache=None, output_final_state=False):
        q, k, v = (tensor.view(-1, self.num_heads, self.head_dim) for tensor in vectors)
        g, beta = self._gates(x)
        return delta_attention(
            q,
            k,
            v,
            g,
            beta,
            prepared,
            cache,
            output_final_state,
            channel_decay=self.channel_decay,
        )

    def _output(self, x, memory):
        gate = self.gate(x).view_as(memory)
        gate = gate.sigmoid() if self.channel_decay else F.silu(gate)
        return self.o((self.norm_output(memory) * gate).reshape_as(x))

    def forward_replay(
        self, x, prepared, previous, cache=None, output_final_state=False
    ):
        """Return the output, reusable K/V/beta, and this loop's temporal cache."""
        memory, histories = (None, (None, None, None)) if cache is None else cache
        if output_final_state:
            if len(prepared[0]) != 2 or (cache is not None and len(x) != 1):
                raise ValueError(
                    "Cached replay requires one document and one new token"
                )
            vectors, new_histories = self._project_cached(x, prepared, histories)
        else:
            vectors = self._project(x, prepared)
        q, k, v = (tensor.view(-1, self.num_heads, self.head_dim) for tensor in vectors)
        g, beta = self._gates(x)
        output, state = replay_delta_attention(
            q,
            k,
            v,
            g,
            beta,
            previous,
            prepared,
            cache=memory,
            output_final_state=output_final_state,
            channel_decay=self.channel_decay,
        )
        new_cache = (state, new_histories) if output_final_state else None
        return self._output(x, output), (k, v, beta), new_cache


class KimiDeltaAttention(GatedDeltaAttention):
    channel_decay = True

    def _project_cached(self, x, prepared, histories):
        if (
            x.device.type != "npu"
            or torch.is_grad_enabled()
            or any(history is None for history in histories)
        ):
            return super()._project_cached(x, prepared, histories)

        from pondera.models.short_conv_npu import short_convolution_step

        projections = self.q(x), self.k(x), self.v(x)
        weights = self.q_conv.weight, self.k_conv.weight, self.v_conv.weight
        fused = short_convolution_step(projections, weights, histories)
        if fused is not None:
            return fused
        return self._convolve_cached(projections, prepared, histories)

    def _prepare_cached(self, x, cache):
        if cache is not None and x.device.type == "npu" and not torch.is_grad_enabled():
            # One sequence of one token uses the batched recurrent kernel.
            return None, None
        return super()._prepare_cached(x, cache)


class GatedDeltaModel(HybridModel):
    architecture = "gdn"
    attention_cls = GatedDeltaAttention


class KimiDeltaModel(GatedDeltaModel):
    architecture = "kda"
    attention_cls = KimiDeltaAttention
