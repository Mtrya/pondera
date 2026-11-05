"""Deterministic, search-free move selection."""

import chess

from pondera.chess.tracker import repetition_count
from pondera.inference.trace import RootPosition, fixed_budget, generate
from pondera.models.policy import PolicyModel


def pick_move(model, board: chess.Board, device="cpu") -> str:
    if not isinstance(model, PolicyModel):
        raise ValueError("Move selection requires the policy model")
    if board.is_game_over():
        return "0000"
    trace = generate(
        model,
        [RootPosition(board.fen(en_passant="fen"), repetition=repetition_count(board))],
        fixed_budget(0),
    )[0]
    return trace.final_move
