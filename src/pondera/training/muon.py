"""Muon (momentum orthogonalized by Newton–Schulz) plus the frozen split.

Muon owns 2D hidden matrices; AdamW owns embeddings, the output head,
norms, biases, and scalars. The split is identical for every architecture.
The update follows Keller Jordan's original Muon: the orthogonalized
update is scaled by sqrt(max(1, fan_out / fan_in)), and the Newton–Schulz
iteration uses the standard five steps.

Hyperball optionally fixes hidden-matrix and update Frobenius norms using
the initial matrix radius (https://arxiv.org/abs/2606.16899).
"""

import torch
from torch import nn


def newton_schulz5(gradient):
    """Orthogonalize a matrix with the quintic Newton–Schulz iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = gradient.bfloat16()
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.mT if transposed else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, *, hyperball=False):
        super().__init__(params, dict(lr=lr, hyperball=hyperball))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param.grad)
                buffer = state["momentum_buffer"]
                buffer.lerp_(param.grad, 1 - 0.95)
                update = newton_schulz5(param.grad.lerp(buffer, 0.95))
                if group["hyperball"]:
                    if "radius" not in state:
                        state["radius"] = param.norm().detach().clone()
                    radius = state["radius"]
                    update = update.to(param.dtype)
                    # Hyperball Eq. (1): normalize the update, then project the
                    # resulting weight back to its initial Frobenius radius.
                    update.mul_(radius / update.norm().clamp_min(1e-30))
                    if lr == 1:
                        # A unit step can cancel the matrix. Zero has no projection.
                        candidate = param - update
                        norm = candidate.norm()
                        projected = candidate * (radius / norm.clamp_min(1e-30))
                        param.copy_(torch.where(norm == 0, param, projected))
                    else:
                        param.add_(update, alpha=-lr)
                        param.mul_(radius / param.norm().clamp_min(1e-30))
                else:
                    scale = max(1, param.shape[-2] / param.shape[-1]) ** 0.5
                    param.add_(update.to(param.dtype), alpha=-lr * scale)
        return loss


def split_parameters(model: nn.Module):
    """Separate trainable hidden matrices from embeddings, head and gains."""
    muon, adamw = [], []
    embeddings = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim == 2
            and id(param) not in embeddings
            and not name.startswith("head.")
        ):
            muon.append(param)
        else:
            adamw.append(param)
    return muon, adamw


def build_optimizers(model, *, optimizer, muon_lr, adamw_lr, adamw_weight_decay):
    muon_params, adamw_params = split_parameters(model)
    return [
        Muon(muon_params, lr=muon_lr, hyperball=optimizer == "hyperball"),
        torch.optim.AdamW(
            adamw_params,
            lr=adamw_lr,
            betas=(0.9, 0.95),
            eps=1e-10,
            weight_decay=adamw_weight_decay,
        ),
    ]
