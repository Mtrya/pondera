"""Sparse board atoms embedded by one joint table, without positional terms."""

from dataclasses import dataclass

import torch
from torch import nn

from pondera.chess.symmetry import TOKEN_MIRROR
from pondera.chess.vocabulary import BOARD_VOCAB_SIZE, PAD, SIDE

_MIRROR = torch.from_numpy(TOKEN_MIRROR)


@dataclass
class BoardEmbeddings:
    hidden: torch.Tensor
    valid: torch.Tensor


def black_to_move(ids):
    """Whether each row carries the black-to-move side atom."""
    return (ids == SIDE + 1).any(-1)


def reflect_board_tokens(ids, reflected):
    """Apply the exact rank-mirror id permutation to the reflected rows."""
    mirrored = _MIRROR.to(ids.device)[ids.long()]
    return torch.where(reflected[:, None], mirrored, ids.long()).to(ids.dtype)


class BoardEmbedding(nn.Module):
    """One row per board or metadata atom, plus a dedicated padding row.

    Padding ids clamp to that row, whose parameters stay zero and receive no
    gradient; padded positions are masked out before attention. Prefix tokens
    carry no positional term: the board is a set of atoms.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.pad_row = BOARD_VOCAB_SIZE
        self.tokens = nn.Embedding(
            BOARD_VOCAB_SIZE + 1, hidden_size, padding_idx=self.pad_row
        )
        nn.init.normal_(self.tokens.weight, std=0.02)
        with torch.no_grad():
            self.tokens.weight[self.pad_row].zero_()

    def forward(self, ids):
        if ids.ndim != 2 or ids.dtype not in {torch.int16, torch.int32, torch.int64}:
            raise ValueError("Expected sparse board token ids")
        ids = reflect_board_tokens(ids, black_to_move(ids))
        valid = ids != PAD
        hidden = self.tokens(ids.long().clamp(max=self.pad_row))
        return BoardEmbeddings(hidden.masked_fill(~valid.unsqueeze(-1), 0), valid)
