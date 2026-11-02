"""Deterministic, search-free move selection."""

import chess

from pondera.chess.symmetry import reflect_uci
from pondera.chess.tracker import repetition_count
from pondera.chess.vocabulary import IDX_TO_UCI_MOVE, MOVE
from pondera.data.trace import CLAIM_DRAW_UCI
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
    if not trace.root_move_tokens:
        return CLAIM_DRAW_UCI
    index = max(
        range(len(trace.root_move_tokens)),
        key=trace.root_move_log_probs.__getitem__,
    )
    uci = IDX_TO_UCI_MOVE[trace.root_move_tokens[index] - MOVE]
    return reflect_uci(uci) if trace.mirrored else uci
