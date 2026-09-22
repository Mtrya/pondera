"""Position history and legal actions."""

import chess

from pondera.chess.mapping import UCI_MOVE_TO_IDX


def repetition_count(board: chess.Board) -> int:
    if board.is_repetition(3):
        return 3
    if board.is_repetition(2):
        return 2
    return 1


def legal_move_indices(board: chess.Board) -> list[int]:
    """Board moves only; draw adjudication belongs to the game runner."""
    return sorted(UCI_MOVE_TO_IDX[move.uci()] for move in board.legal_moves)
