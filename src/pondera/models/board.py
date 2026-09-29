"""Square and compact-piece inputs, shared by all token backbones."""

from dataclasses import dataclass

import torch
from torch import nn

from pondera.chess.board_tokens import (
    CASTLING,
    EN_PASSANT,
    MIN_TOKENS,
    PAD,
    SIDE,
    VOCAB_SIZE,
)
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


def black_to_move(ids, positions=None):
    if positions is None:
        return ids[:, 64] == SIDE + 1
    return ((positions == 64) & (ids == SIDE + 1)).any(-1)


def reflect_board_tokens(ids, positions, reflected):
    """Reflect ranks and swap colors while preserving the original side flag.

    The unchanged flag records the original orientation, so this transform is
    reversible. Castling fields and en-passant coordinates change with the board;
    counters and supplied repetition information retain their exact values.
    """
    valid = ids != PAD
    board = (positions < 64) & valid
    castling = (positions >= 65) & (positions < 69) & valid
    ep = (positions == 69) & valid & (ids != EN_PASSANT + 64)
    new_positions = torch.where(board, positions ^ 56, positions)
    new_positions = torch.where(castling, 65 + ((positions - 65) ^ 2), new_positions)
    new_ids = torch.where(board & (ids < EMPTY_SQ_IDX), (ids + 6) % 12, ids)
    new_ids = torch.where(
        castling, CASTLING + 2 * (new_positions - 65) + ((ids - CASTLING) % 2), new_ids
    )
    new_ids = torch.where(ep, EN_PASSANT + ((ids - EN_PASSANT) ^ 56), new_ids)
    return torch.where(reflected[:, None], new_ids, ids).to(ids.dtype), torch.where(
        reflected[:, None], new_positions, positions
    )


class BoardEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size,
        layout="squares",
        encoding="additive",
        orientation="absolute",
        geometry="absolute",
    ):
        super().__init__()
        if layout not in {"squares", "pieces"}:
            raise ValueError("Unknown board layout")
        if encoding not in {"additive", "joint"} or orientation not in {
            "absolute",
            "side_to_move",
        }:
            raise ValueError("Unknown board encoding or orientation")
        if geometry not in {"absolute", "relative", "sin2d", "rope2d"}:
            raise ValueError("Unknown board geometry")
        if encoding == "joint" and geometry in {"sin2d", "rope2d"}:
            raise ValueError("Joint binding carries its own square identity")
        self.layout = layout
        self.encoding = encoding
        self.orientation = orientation
        self.geometry = geometry
        offset = 13 if encoding == "joint" else 0
        self.tokens = nn.Embedding(
            VOCAB_SIZE - offset, hidden_size, padding_idx=PAD - offset
        )
        if encoding == "joint" or geometry in {"absolute", "relative"}:
            self.squares = nn.Embedding(
                13 * 64 if encoding == "joint" else 64, hidden_size
            )
        elif geometry == "sin2d":
            files = sinusoidal_positions(
                torch.arange(8), (hidden_size + 1) // 2, torch.float32
            )
            ranks = sinusoidal_positions(
                torch.arange(8), hidden_size // 2, torch.float32
            )
            grid = torch.cat(
                (
                    files[:, None, :].expand(8, 8, -1),
                    ranks[None, :, :].expand(8, 8, -1),
                ),
                dim=-1,
            )[:, :, :hidden_size]
            self.register_buffer(
                "squares_fixed", grid.reshape(64, hidden_size), persistent=False
            )
        nn.init.normal_(self.tokens.weight, std=0.02)
        if hasattr(self, "squares"):
            nn.init.normal_(self.squares.weight, std=0.02)
        with torch.no_grad():
            self.tokens.weight[PAD - offset].zero_()

    def forward(self, ids, positions=None):
        if ids.ndim != 2 or ids.dtype not in {torch.int16, torch.int32, torch.int64}:
            raise ValueError("Expected lossless integer board tokens")
        if positions is None:
            if ids.shape[1] < MIN_TOKENS:
                raise ValueError(
                    "Unpacked inputs must contain a complete lossless board"
                )
            packed = pack_board_tokens(ids, self.layout)
            ids, positions = packed["ids"], packed["positions"]
        if self.orientation == "side_to_move":
            ids, positions = reflect_board_tokens(
                ids, positions, black_to_move(ids, positions)
            )
        valid = ids != PAD
        board = (positions < 64) & valid
        squares = positions.masked_fill(~board, -1)
        if self.encoding == "joint":
            content = self.tokens((ids.long() - 13).clamp_min(0))
            spatial = self.squares(
                (squares.clamp_min(0) * 13 + ids.long().clamp(0, 12))
            )
        elif self.geometry == "sin2d":
            content = self.tokens(ids.long())
            spatial = self.squares_fixed[squares.clamp_min(0)].to(content.dtype)
        elif self.geometry == "rope2d":
            content = self.tokens(ids.long())
            spatial = 0
        else:
            content = self.tokens(ids.long())
            spatial = self.squares(squares.clamp_min(0))
        metadata_position = sinusoidal_positions(
            (positions - 64).clamp_min(0), content.shape[-1], content.dtype
        )
        hidden = (
            torch.where(board.unsqueeze(-1), spatial, content + metadata_position)
            if self.encoding == "joint"
            else content + torch.where(board.unsqueeze(-1), spatial, metadata_position)
        )
        return BoardEmbeddings(
            hidden.masked_fill(~valid.unsqueeze(-1), 0), valid, squares
        )
