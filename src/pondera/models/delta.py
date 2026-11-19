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


class GatedDeltaAttention(RecurrentAttention):
    channel_decay = False

    def __init__(self, hidden_size, num_heads, num_kv_heads):
        # Recurrent layers use full memory heads; num_kv_heads applies to softmax.
        super().__init__(hidden_size, num_heads)
        self.beta = nn.Linear(hidden_size, num_heads, bias=False)
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
        cu_cpu, cu_device = prepared
        if x.device.type in ("cuda", "npu"):
            if self.channel_decay:
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
            incremental = cache is not None or (
                not torch.is_grad_enabled() and q.dtype == torch.float32
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
                use_qk_l2norm_in_kernel=True,
                **options,
            )
            return output.squeeze(0), state
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

    def _output(self, x, memory):
        gate = self.gate(x).view_as(memory)
        gate = gate.sigmoid() if self.channel_decay else F.silu(gate)
        return self.o((self.norm_output(memory) * gate).reshape_as(x))


class KimiDeltaAttention(GatedDeltaAttention):
    channel_decay = True


class GatedDeltaModel(HybridModel):
    architecture = "gdn"
    attention_cls = GatedDeltaAttention


class KimiDeltaModel(GatedDeltaModel):
    architecture = "kda"
    attention_cls = KimiDeltaAttention
