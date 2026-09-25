"""Repetition-indexed KV memory with execution-order-independent access rules."""

from dataclasses import dataclass

import torch

CACHE_POLICIES = {"matched_fallback", "first", "matched_only"}


@dataclass(frozen=True)
class KVState:
    key: torch.Tensor
    value: torch.Tensor
    valid: torch.Tensor

    def detach(self):
        return KVState(self.key.detach(), self.value.detach(), self.valid)


def select_state(history: list[KVState], repetition: int, policy: str) -> KVState:
    """Read only entries at or before a query's repetition, never a future exit."""
    if policy == "first":
        return history[0]
    if policy == "matched_only":
        if repetition < len(history):
            return history[repetition]
        first = history[0]
        return KVState(first.key, first.value, torch.zeros_like(first.valid))
    if policy != "matched_fallback":
        raise ValueError(f"Unknown cache policy: {policy}")
    selected = history[0]
    for entry in history[1 : repetition + 1]:
        mask = entry.valid[:, None, :, None]
        selected = KVState(
            torch.where(mask, entry.key, selected.key),
            torch.where(mask, entry.value, selected.value),
            selected.valid | entry.valid,
        )
    return selected


def join_states(states: list[KVState]) -> KVState:
    if len(states) == 1:
        return states[0]
    return KVState(
        torch.cat([s.key for s in states], dim=2),
        torch.cat([s.value for s in states], dim=2),
        torch.cat([s.valid for s in states], dim=1),
    )


@dataclass(frozen=True)
class BackboneCache:
    owner: int
    prefix_length: int
    squares: torch.Tensor
    # Physical (core, layer) -> chunks -> executed repetitions within each chunk.
    histories: dict[tuple[int, int], list[list[KVState]]]

    @property
    def length(self):
        return self.squares.shape[1]
