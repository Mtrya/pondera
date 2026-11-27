"""Attention over groups of residual updates at the same token."""

import torch
from torch import nn
from torch.nn import functional as F


def summarize(value):
    value = value.float()
    inverse_rms = torch.rsqrt(value.square().mean(dim=-1) + 1e-6)
    return value, inverse_rms


class DepthRead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(width))
        self.key_gain = nn.Parameter(torch.ones(width))
        self.internals = None
        self.internals_position = 0

    def forward(self, residual, sources):
        query = (self.query * self.key_gain).float().unsqueeze(0)
        if self.training and query.device.type == "npu" and len(sources) <= 6:
            from pondera.models._depth_read_triton import depth_read

            # The group increments telescope to the residual. Scaling the mean by
            # the number of sources gives the same read with a fused backward.
            return len(sources) * depth_read(sources, query)
        with torch.autocast(query.device.type, enabled=False):
            scores = torch.stack(
                [
                    F.linear(value, query).squeeze(-1) * inverse_rms
                    for value, inverse_rms in sources
                ],
                dim=-1,
            )
        probs = scores.softmax(dim=-1)
        if self.internals is not None:
            key = f"read/pos{self.internals_position}/pass{self.internals.pass_index}"
            # The read concentrates when one source's probability dominates;
            # a uniform read of n sources reports n effective sources.
            self.internals.mean(f"{key}/max_prob", probs.max(dim=-1).values)
            self.internals.mean(f"{key}/effective", probs.square().sum(-1).reciprocal())
        # Unit weights reconstruct the trained residual with its original addition order.
        weights = (len(sources) * probs - 1).unbind(-1)
        for (value, _), weight in zip(sources, weights, strict=True):
            residual = torch.addcmul(residual, value, weight.unsqueeze(-1))
        return residual
