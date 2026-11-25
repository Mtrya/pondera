"""Frozen affine/ReLU circuits for chess piece-state transitions."""

import torch
from torch import nn

from pondera.chess.vocabulary import IDX_TO_UCI_MOVE, MOVES


def _frozen_linear(weights, bias=None):
    weights = torch.tensor(weights, dtype=torch.float32)
    layer = nn.Linear(weights.shape[1], weights.shape[0], bias=bias is not None)
    layer.weight = nn.Parameter(weights, requires_grad=False)
    if bias is not None:
        layer.bias = nn.Parameter(
            torch.tensor(bias, dtype=torch.float32), requires_grad=False
        )
    return layer


class CompiledPieceTransition(nn.Module):
    """Update square/piece indicators for a qualified legal move.

    Input pieces have shape (..., 64, 12), in PNBRQKpnbrqk order; moves
    are indices in the existing move vocabulary. This circuit implements
    piece placement only. Legality, counters and path navigation belong to
    its caller. The draw-claim entry preserves placement.
    """

    def __init__(self):
        super().__init__()
        # Five square selectors, followed by the promoted piece indicator.
        table = torch.zeros(MOVES, 5 * 64 + 12)
        castles = {"e1g1": (7, 5), "e1c1": (0, 3), "e8g8": (63, 61), "e8c8": (56, 59)}
        for index, move in IDX_TO_UCI_MOVE.items():
            if move == "<claim_draw>":
                continue
            source = ord(move[0]) - ord("a") + 8 * (int(move[1]) - 1)
            target = ord(move[2]) - ord("a") + 8 * (int(move[3]) - 1)
            table[index, source] = 1
            table[index, 64 + target] = 1
            if (
                abs(target // 8 - source // 8) == 1
                and abs(target % 8 - source % 8) == 1
            ):
                table[index, 128 + source // 8 * 8 + target % 8] = 1
            if move in castles:
                rook_source, rook_target = castles[move]
                table[index, 192 + rook_source] = 1
                table[index, 256 + rook_target] = 1
            if len(move) == 5:
                color = 0 if target // 8 == 7 else 1
                table[index, 320 + 6 * color + "pnbrqk".index(move[4])] = 1
        self.move_embedding = nn.Embedding.from_pretrained(table, freeze=True)
        self.conjunction = _frozen_linear([[1, 1]], [-1])
        self.erase = _frozen_linear([[1, -1]])
        self.sum_squares = _frozen_linear([[1] * 64])

    def _and(self, left, right):
        left, right = torch.broadcast_tensors(left, right)
        return self.conjunction(torch.stack((left, right), dim=-1)).squeeze(-1).relu()

    def _read(self, pieces, selector):
        selected = self._and(pieces, selector.unsqueeze(-1))
        return self.sum_squares(selected.transpose(-1, -2)).squeeze(-1)

    def forward(self, pieces, moves):
        source, target, ep_square, rook_source, rook_target, promotion = (
            self.move_embedding(moves).split((64, 64, 64, 64, 64, 12), dim=-1)
        )
        moving = self._read(pieces, source)
        captured = self._read(pieces, target).sum(-1, keepdim=True)
        pawn = moving[..., 0:1] + moving[..., 6:7]
        king = moving[..., 5:6] + moving[..., 11:12]
        en_passant = self._and(pawn, 1 - captured)
        rook_source = self._and(rook_source, king)
        rook_target = self._and(rook_target, king)
        rook = self._read(pieces, rook_source)
        removed = source + target + rook_source + self._and(ep_square, en_passant)
        retained = (
            self.erase(
                torch.stack((pieces, removed.unsqueeze(-1).expand_as(pieces)), dim=-1)
            )
            .squeeze(-1)
            .relu()
        )
        promoted = promotion.sum(-1, keepdim=True)
        arriving = self._and(moving, 1 - promoted) + promotion
        return (
            retained
            + self._and(arriving.unsqueeze(-2), target.unsqueeze(-1))
            + self._and(rook.unsqueeze(-2), rook_target.unsqueeze(-1))
        )
