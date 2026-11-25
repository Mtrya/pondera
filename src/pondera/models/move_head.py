"""Shared source and destination factors for the atomic move logits."""

import chess
import torch
from torch import nn
from torch.nn import functional as F

from pondera.chess.vocabulary import IDX_TO_UCI_MOVE, MOVE, VALUE


class FactorizedMoveHead(nn.Linear):
    """Keep a distinct weight per token and share factors among board moves."""

    def __init__(self, hidden_size, vocab_size):
        if vocab_size < VALUE:
            raise ValueError(
                "A factorized move head requires the complete move vocabulary"
            )
        super().__init__(hidden_size, vocab_size, bias=False)
        self.source = nn.Parameter(torch.zeros(64, hidden_size))
        self.destination = nn.Parameter(torch.zeros(64, hidden_size))
        coordinates = [
            (MOVE + index, chess.parse_square(move[:2]), chess.parse_square(move[2:4]))
            for index, move in IDX_TO_UCI_MOVE.items()
            if move != "<claim_draw>"
        ]
        for name, indices in zip(
            ("move_ids", "sources", "destinations"), zip(*coordinates), strict=True
        ):
            self.register_buffer(name, torch.tensor(indices), persistent=False)

    def forward(self, hidden):
        factors = self.source[self.sources] + self.destination[self.destinations]
        weight = torch.index_add(self.weight, 0, self.move_ids, factors)
        return F.linear(hidden, weight)
