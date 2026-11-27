"""Packed short convolutions and incremental state for recurrent attention."""

import torch
from torch import nn
from torch.nn import functional as F


class CausalConvolution(nn.Conv1d):
    """Four-token depthwise convolution, reset at each document."""

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
            tail = x[-3:]
            # Keep the tail without retaining the full prefill projection.
            history = (
                F.pad(tail, (0, 0, 3 - tail.shape[0], 0))
                if tail.shape[0] < 3
                else tail.clone()
            )
        else:
            window = torch.cat((cache, x), dim=0)
            result = F.silu((window.T * self.weight.squeeze(1)).sum(-1)).unsqueeze(0)
            history = window[-3:]
        return result, history


class RecurrentAttention(nn.Module):
    """Q/K/V convolutions with a subclass's memory update and output readout."""

    @staticmethod
    def prepare(cu_seqlens, device):
        cu = torch.as_tensor(cu_seqlens, dtype=torch.int64, device="cpu")
        return cu, cu.to(device, non_blocking=True)

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("The head count must divide the hidden width")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_conv = CausalConvolution(hidden_size)
        self.k_conv = CausalConvolution(hidden_size)
        self.v_conv = CausalConvolution(hidden_size)

    def _project(self, x, prepared):
        return tuple(
            conv(projection(x), prepared)
            for projection, conv in (
                (self.q, self.q_conv),
                (self.k, self.k_conv),
                (self.v, self.v_conv),
            )
        )

    def _project_cached(self, x, prepared, histories):
        projections = (projection(x) for projection in (self.q, self.k, self.v))
        return self._convolve_cached(projections, prepared, histories)

    def _convolve_cached(self, projections, prepared, histories):
        vectors, new_histories = [], []
        for projected, conv, history in zip(
            projections, (self.q_conv, self.k_conv, self.v_conv), histories
        ):
            vector, tail = conv.forward_cached(projected, prepared, history)
            vectors.append(vector)
            new_histories.append(tail)
        return vectors, tuple(new_histories)

    def forward(self, x, prepared):
        output, _ = self._memory(self._project(x, prepared), x, prepared)
        return self._output(x, output)

    def _prepare_cached(self, x, cache):
        return self.prepare([0, x.shape[0]], x.device)

    def forward_cached(self, x, cu_seqlens, cache=None):
        if cache is not None and x.shape[0] != 1:
            raise ValueError("Cached memory decoding consumes one new token")
        if len(cu_seqlens) != 2:
            raise ValueError("Cached decoding requires one document")
        prepared = self._prepare_cached(x, cache)
        memory, histories = (None, (None, None, None)) if cache is None else cache
        vectors, new_histories = self._project_cached(x, prepared, histories)
        output, state = self._memory(vectors, x, prepared, memory, True)
        return self._output(x, output), (state, new_histories)
