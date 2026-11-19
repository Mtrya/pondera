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

from pondera.models.vanilla import CausalSelfAttention, RMSNorm, VanillaModel


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


class CausalConvolution(nn.Conv1d):
    """Four-token depthwise SiLU convolution, reset at each document."""

    def __init__(self, width):
        super().__init__(width, width, kernel_size=4, groups=width, bias=False)

    def forward(self, x, prepared):
        cu_cpu, cu_device = prepared
        if x.device.type in ("cuda", "npu"):
            from fla.modules.causal_conv1d import causal_conv1d

            output, _ = causal_conv1d(
                x.unsqueeze(0),
                self.weight.squeeze(1),
                activation="silu",
                cu_seqlens=cu_device,
                cu_seqlens_cpu=cu_cpu,
            )
            return output.squeeze(0)
        pieces = []
        for start, stop in zip(cu_cpu[:-1].tolist(), cu_cpu[1:].tolist()):
            doc = F.pad(x[start:stop].T.unsqueeze(0), (3, 0))
            pieces.append(F.silu(super().forward(doc)).squeeze(0).T)
        return torch.cat(pieces)

    def forward_cached(self, x, prepared, cache=None):
        if cache is None:
            result = self(x, prepared)
            history = F.pad(x, (0, 0, max(0, 3 - x.shape[0]), 0))[-3:]
        else:
            window = torch.cat((cache, x), dim=0)
            result = F.silu((window.T * self.weight.squeeze(1)).sum(-1)).unsqueeze(0)
            history = window[-3:]
        return result, history


class GatedDeltaAttention(nn.Module):
    channel_decay = False

    @staticmethod
    def prepare(cu_seqlens, device):
        cu = torch.as_tensor(cu_seqlens, dtype=torch.int64, device="cpu")
        return cu, cu.to(device, non_blocking=True)

    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("The head count must divide the hidden width")
        # Delta layers use one memory per head. num_kv_heads controls only the
        # softmax layers when this mixer is used in a hybrid model.
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_conv = CausalConvolution(hidden_size)
        self.k_conv = CausalConvolution(hidden_size)
        self.v_conv = CausalConvolution(hidden_size)
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

    def forward(self, x, prepared):
        vectors = (
            conv(projection(x), prepared)
            for projection, conv in (
                (self.q, self.q_conv),
                (self.k, self.k_conv),
                (self.v, self.v_conv),
            )
        )
        output, _ = self._memory(vectors, x, prepared)
        return self._output(x, output)

    def forward_cached(self, x, cu_seqlens, cache=None):
        if cache is not None and x.shape[0] != 1:
            raise ValueError("Cached delta decoding consumes one new token")
        if len(cu_seqlens) != 2:
            raise ValueError("Cached decoding requires one document")
        prepared = self.prepare([0, x.shape[0]], x.device)
        memory, histories = (None, (None, None, None)) if cache is None else cache
        vectors, new_histories = [], []
        for projection, conv, history in zip(
            (self.q, self.k, self.v), (self.q_conv, self.k_conv, self.v_conv), histories
        ):
            vector, tail = conv.forward_cached(projection(x), prepared, history)
            vectors.append(vector)
            new_histories.append(tail)
        output, state = self._memory(vectors, x, prepared, memory, True)
        return self._output(x, output), (state, tuple(new_histories))


class KimiDeltaAttention(GatedDeltaAttention):
    channel_decay = True


class GatedDeltaModel(VanillaModel):
    architecture = "gdn"
    attention_cls = GatedDeltaAttention

    def __init__(self, *, softmax_every=0, **kwargs):
        if not 0 <= softmax_every <= kwargs["num_blocks"]:
            raise ValueError("softmax_every must be zero or a valid layer interval")
        if not softmax_every and kwargs["num_kv_heads"] != kwargs["num_heads"]:
            raise ValueError("Pure delta models require num_kv_heads = num_heads")
        self.softmax_every = softmax_every
        super().__init__(**kwargs)

    def attention_type(self, index):
        if self.softmax_every and (index + 1) % self.softmax_every == 0:
            return CausalSelfAttention
        return self.attention_cls


class KimiDeltaModel(GatedDeltaModel):
    architecture = "kda"
    attention_cls = KimiDeltaAttention
