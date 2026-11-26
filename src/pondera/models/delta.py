"""Kimi Delta Attention with a packed kernel and an FP32 recurrence."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from pondera.models.memory import RecurrentAttention
from pondera.models.softmax import RMSNorm


def delta_recurrent(q, k, v, g, beta, initial_state=None):
    """One document, [T,H,D] vectors and fp32 [H,D,D] memory."""
    q, k, v, g, beta = (tensor.float() for tensor in (q, k, v, g, beta))
    state = (
        q.new_zeros(q.shape[1], q.shape[2], v.shape[2])
        if initial_state is None
        else initial_state.float()
    )
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
):
    """Run KDA over packed documents or a cached token."""
    cu_cpu, cu_device = prepared
    if q.device.type in ("cuda", "npu"):
        from fla.ops.kda import chunk_kda, fused_recurrent_kda

        incremental = not torch.is_grad_enabled() and (
            cache is not None or q.dtype == torch.float32
        )
        options = {} if incremental else {"cu_seqlens_cpu": cu_cpu}
        output, state = (fused_recurrent_kda if incremental else chunk_kda)(
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


class KimiDeltaAttention(RecurrentAttention):
    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__(hidden_size, num_heads)
        self.beta = nn.Linear(hidden_size, num_heads, bias=False)
        self.decay = nn.Sequential(
            nn.Linear(hidden_size, self.head_dim, bias=False),
            nn.Linear(self.head_dim, hidden_size, bias=False),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, self.head_dim, bias=False),
            nn.Linear(self.head_dim, hidden_size, bias=True),
        )
        self.A_log = nn.Parameter(torch.empty(num_heads).uniform_(1.0, 16).log())
        dt = torch.exp(
            torch.empty(hidden_size).uniform_(math.log(0.001), math.log(0.1))
        )
        self.dt_bias = nn.Parameter(dt + (-torch.expm1(-dt)).log())
        self.norm_output = RMSNorm(self.head_dim, eps=1e-5)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.internals = None
        self.internals_name = "lay?"

    def _memory(self, vectors, x, prepared, cache=None, output_final_state=False):
        q, k, v = (tensor.view(-1, self.num_heads, self.head_dim) for tensor in vectors)
        decay = F.softplus(self.decay(x).float() + self.dt_bias).view_as(q)
        decay = -self.A_log.exp()[None, :, None] * decay
        beta = self.beta(x).float().sigmoid()
        if self.internals is not None:
            name = f"kda/{self.internals_name}/pass{self.internals.pass_index}"
            self.internals.mean(f"{name}/beta", beta)
            self.internals.mean(f"{name}/retention", decay.exp())
        return delta_attention(
            q, k, v, decay, beta, prepared, cache, output_final_state
        )

    def _output(self, x, memory):
        gate = self.gate(x).view_as(memory).sigmoid()
        if self.internals is not None:
            name = f"kda/{self.internals_name}/pass{self.internals.pass_index}"
            self.internals.mean(f"{name}/gate", gate)
        return self.o((self.norm_output(memory) * gate).reshape_as(x))

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
