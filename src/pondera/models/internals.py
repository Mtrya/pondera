"""Model-internal measurements from one validation batch."""

import torch


class Internals:
    def __init__(self):
        self.pass_index = 1
        self._scalars = {}
        self._token_rms = {}

    def mean(self, key, value):
        self._scalars[key] = value.detach().float().mean()

    def rms(self, key, value, *, token_tail=False):
        square = value.detach().float().square()
        if token_tail:
            per_token = square.mean(dim=-1)
            self._token_rms[key] = per_token.sqrt()
            self._scalars[key] = per_token.mean().sqrt()
        else:
            self._scalars[key] = square.mean().sqrt()

    def report(self):
        if not self._scalars:
            return {}
        result = dict(
            zip(
                self._scalars,
                torch.stack(tuple(self._scalars.values())).tolist(),
                strict=True,
            )
        )
        for key, values in self._token_rms.items():
            result[f"{key}/token_rms_p99"] = float(torch.quantile(values.cpu(), 0.99))
        return result


def attach(model, internals):
    """Collect the next model forward, including its depth and KDA reads."""
    model.internals = internals
    for index, read in enumerate(model.depth_reads, 1):
        read.internals = internals
        read.internals_position = index
    for index, block in enumerate(model.blocks, 1):
        block.attention.internals = internals
        block.attention.internals_name = f"lay{index:02d}"


def detach(model):
    model.internals = None
    for read in model.depth_reads:
        read.internals = None
    for block in model.blocks:
        block.attention.internals = None
