"""Square and compact-piece inputs, shared by all token backbones."""

from dataclasses import dataclass

import torch
from torch import nn

from pondera.chess.board_tokens import MIN_TOKENS, PAD, VOCAB_SIZE
from pondera.chess.mapping import EMPTY_SQ_IDX
from pondera.models.blocks import sinusoidal_positions


def pack_board_tokens(ids, layout="squares"):
    """Prepare a CPU batch before device transfer; retain original square IDs."""
    if layout not in {"squares", "pieces"}:
        raise ValueError("Unknown board layout")
    positions = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
    keep = ids != PAD
    if layout == "pieces":
        keep = keep & ((positions >= 64) | (ids != EMPTY_SQ_IDX))
    width = int(keep.sum(1).max())
    order = torch.argsort(~keep, dim=1, stable=True)[:, :width]
    valid = keep.gather(1, order)
    return {
        "ids": ids.gather(1, order).masked_fill(~valid, PAD),
        "positions": positions.gather(1, order),
    }


@dataclass
class BoardEmbeddings:
    hidden: torch.Tensor
    valid: torch.Tensor
    squares: torch.Tensor


class BoardEmbedding(nn.Module):
    def __init__(self, hidden_size, layout="squares"):
        super().__init__()
        if layout not in {"squares", "pieces"}:
            raise ValueError("Unknown board layout")
        self.layout = layout
        self.tokens = nn.Embedding(VOCAB_SIZE, hidden_size, padding_idx=PAD)
        self.squares = nn.Embedding(64, hidden_size)
        nn.init.normal_(self.tokens.weight, std=0.02)
        nn.init.normal_(self.squares.weight, std=0.02)
        with torch.no_grad():
            self.tokens.weight[PAD].zero_()

    def forward(self, ids, positions=None):
        if ids.ndim != 2 or ids.dtype not in {torch.int16, torch.int32, torch.int64}:
            raise ValueError(
                "Expected lossless integer board tokens, not legacy uint8 inputs"
            )
        if positions is None:
            if ids.shape[1] < MIN_TOKENS:
                raise ValueError(
                    "Unpacked inputs must contain a complete lossless board"
                )
            packed = pack_board_tokens(ids, self.layout)
            ids, positions = packed["ids"], packed["positions"]
        valid = ids != PAD
        board = (positions < 64) & valid
        squares = positions.masked_fill(~board, -1)
        content = self.tokens(ids.long())
        spatial = self.squares(squares.clamp_min(0))
        metadata_position = sinusoidal_positions(
            (positions - 64).clamp_min(0), content.shape[-1], content.dtype
        )
        hidden = content + torch.where(board.unsqueeze(-1), spatial, metadata_position)
        return BoardEmbeddings(
            hidden.masked_fill(~valid.unsqueeze(-1), 0), valid, squares
        )
