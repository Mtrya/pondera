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
        weights = (len(sources) * scores.softmax(dim=-1) - 1).unbind(-1)
        # Unit weights reconstruct the trained residual with its original addition order.
        for (value, _), weight in zip(sources, weights, strict=True):
            residual = torch.addcmul(residual, value, weight.unsqueeze(-1))
        return residual
