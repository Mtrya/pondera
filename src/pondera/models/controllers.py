"""Deployment repetition decisions; objectives for learned decisions are external."""

from dataclasses import dataclass

import torch
from torch import nn

from pondera.models.config import RepetitionConfig


@dataclass
class Decision:
    stop: torch.Tensor
    logits: torch.Tensor | None = None


def group_mean(values, group):
    weights = group.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(1, keepdim=True) / weights.sum(
        1, keepdim=True
    ).clamp_min(1)


class FixedController(nn.Module):
    def forward(self, previous, current, shared_group):
        return Decision(
            torch.zeros(current.shape[:2], dtype=torch.bool, device=current.device)
        )


class ConvergenceController(nn.Module):
    def __init__(self, tolerance):
        super().__init__()
        self.tolerance = tolerance

    def forward(self, previous, current, shared_group):
        delta = (current.float() - previous.float()).square().mean(-1).sqrt()
        scale = previous.float().square().mean(-1).sqrt().clamp_min(1e-6)
        stop = delta / scale <= self.tolerance
        # A shared group converges only when every member converges.
        shared_stop = (stop | ~shared_group).all(1, keepdim=True)
        return Decision(torch.where(shared_group, shared_stop, stop))


class LearnedController(nn.Module):
    def __init__(self, hidden_size, threshold):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, 1)
        self.threshold = threshold

    def forward(self, previous, current, shared_group):
        normalized = self.norm(current)
        inputs = torch.where(
            shared_group.unsqueeze(-1), group_mean(normalized, shared_group), normalized
        )
        logits = self.head(inputs).squeeze(-1)
        return Decision(logits.sigmoid() >= self.threshold, logits)


def make_controller(config: RepetitionConfig, hidden_size):
    if config.mode == "fixed":
        return FixedController()
    if config.mode == "convergence":
        return ConvergenceController(config.tolerance)
    return LearnedController(hidden_size, config.stop_threshold)
