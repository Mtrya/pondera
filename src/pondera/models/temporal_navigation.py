"""Content-dependent reads from two causal positions after BACK or RESET."""

from copy import copy
from dataclasses import dataclass

import torch
from torch import nn

from pondera.models.navigation import BranchEntries


@dataclass(frozen=True)
class NavigationCache:
    position: int
    cursor: BranchEntries
    hidden: dict[int, torch.Tensor]


class NavigationRead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(width))
        self.score_scale = width**-0.5

    def _mix(self, current, sources):
        states = torch.cat((current[:, None], sources), dim=1).float()
        keys = states * torch.rsqrt(states.square().mean(dim=-1, keepdim=True) + 1e-6)
        query = keys[:, :1] * self.query.float() * self.score_scale
        scores = (keys * query).sum(dim=-1)
        weights = (3 * scores.softmax(dim=-1)[:, 1:] - 1).unsqueeze(-1)
        return ((states[:, 1:] - states[:, :1]) * weights).sum(dim=1).to(current.dtype)

    def _add(self, hidden, bank, receivers, sources):
        receiver_indices = torch.as_tensor(
            receivers, dtype=torch.long, device=hidden.device
        )
        sender_indices = torch.as_tensor(
            sources, dtype=torch.long, device=hidden.device
        )
        current = hidden.index_select(0, receiver_indices)
        previous = bank.index_select(0, sender_indices.flatten()).unflatten(0, (-1, 2))
        return hidden.index_add(0, receiver_indices, self._mix(current, previous))

    def forward(self, hidden, prepared):
        receivers, sources = prepared
        if not len(receivers):
            return hidden + self.query.sum().to(hidden.dtype) * 0
        return self._add(hidden, hidden, receivers, sources)

    def forward_cached(self, hidden, tokens, table, cache=None):
        if cache is None:
            cursor, start, prior = BranchEntries(), 0, {}
        else:
            cursor, start, prior = copy(cache.cursor), cache.position, cache.hidden
            cursor.path = cursor.path.copy()
        if len(hidden) != len(tokens):
            raise ValueError("Expected one layer output per input token")
        references = [
            cursor.consume(start + i, table[token]) for i, token in enumerate(tokens)
        ]
        receivers = [i for i, (branch, _) in enumerate(references) if branch >= 0]
        if not receivers:
            output = hidden
        elif len(hidden) == 1:
            earlier = torch.stack(
                [prior[source] for source in references[0]]
            ).unsqueeze(0)
            output = hidden + self._mix(hidden, earlier)
        else:
            bank = (
                torch.cat((hidden, torch.stack(tuple(prior.values()))))
                if prior
                else hidden
            )
            previous = {source: len(hidden) + i for i, source in enumerate(prior)}
            senders = [
                [
                    source - start if source >= start else previous[source]
                    for source in references[i]
                ]
                for i in receivers
            ]
            output = self._add(hidden, bank, receivers, senders)
        needed = (
            set()
            if cursor.final
            else {cursor.root, cursor.latest_move, *cursor.path} - {-1}
        )
        saved = {
            index: prior[index] if index < start else hidden[index - start].clone()
            for index in needed
        }
        return output, NavigationCache(start + len(tokens), cursor, saved)
