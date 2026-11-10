"""Muon (momentum orthogonalized by Newton–Schulz) plus the frozen split.

Muon owns 2D hidden matrices; AdamW owns embeddings, the output head,
norms, biases, and scalars. The split is identical for every architecture.
The update follows Keller Jordan's original Muon: the orthogonalized
update is scaled by sqrt(max(1, fan_out / fan_in)), and the Newton–Schulz
iteration uses the standard five steps.
"""

import torch
from torch import nn


def newton_schulz5(gradient, steps=5):
    """Orthogonalize a matrix with the quintic Newton–Schulz iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = gradient.bfloat16()
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.mT if transposed else x


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.0,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param.grad)
                buffer = state["momentum_buffer"]
                buffer.lerp_(param.grad, 1 - group["momentum"])
                gradient = (
                    param.grad.lerp(buffer, group["momentum"])
                    if group["nesterov"]
                    else buffer
                )
                update = newton_schulz5(gradient, group["ns_steps"])
                scale = max(1, param.shape[-2] / param.shape[-1]) ** 0.5
                param.mul_(1 - group["lr"] * group["weight_decay"])
                param.add_(update.to(param.dtype), alpha=-group["lr"] * scale)
        return loss


def split_parameters(model: nn.Module):
    """Return (muon_params, adamw_params) under the frozen split rule."""
    muon, adamw = [], []
    for name, param in model.named_parameters():
        if param.ndim == 2 and not name.startswith(("token_embedding.", "head.")):
            muon.append(param)
        else:
            adamw.append(param)
    return muon, adamw


def build_optimizers(model, *, muon_lr=0.02, adamw_lr=3e-4, weight_decay=0.0):
    muon_params, adamw_params = split_parameters(model)
    return [
        Muon(muon_params, lr=muon_lr, weight_decay=weight_decay),
        torch.optim.AdamW(
            adamw_params,
            lr=adamw_lr,
            betas=(0.9, 0.95),
            eps=1e-10,
            weight_decay=weight_decay,
        ),
    ]
