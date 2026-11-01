"""Key/value state for incremental trace decoding."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KVState:
    key: torch.Tensor
    value: torch.Tensor
    valid: torch.Tensor

    def detach(self):
        return KVState(self.key.detach(), self.value.detach(), self.valid)


def join_states(states: list[KVState]) -> KVState:
    if len(states) == 1:
        return states[0]
    return KVState(
        torch.cat([state.key for state in states], dim=2),
        torch.cat([state.value for state in states], dim=2),
        torch.cat([state.valid for state in states], dim=1),
    )


@dataclass(frozen=True)
class BackboneCache:
    owner: int
    prefix_length: int
    length: int
    histories: dict[int, list[KVState]]
