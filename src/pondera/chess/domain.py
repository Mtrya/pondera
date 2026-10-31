"""Chess domain primitives: pieces, squares, and position history.

This module knows the game but no token ids; everything id-shaped lives in
`pondera.chess.vocabulary` and `pondera.chess.codec`.
"""

from typing import Dict

import chess

PIECE_TO_IDX: Dict[str, int] = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
    ".": 12,
}
IDX_TO_PIECE: Dict[int, str] = {v: k for k, v in PIECE_TO_IDX.items()}
EMPTY_SQ_IDX = PIECE_TO_IDX["."]
# Map algebraic square notation (e.g., 'a1', 'h8') to 0-63 index
# a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
SQUARE_TO_IDX: Dict[str, int] = {
    f"{file}{rank}": (rank - 1) * 8 + (ord(file) - ord("a"))
    for rank in range(1, 9)
    for file in "abcdefgh"
}
IDX_TO_SQUARE: Dict[int, str] = {v: k for k, v in SQUARE_TO_IDX.items()}


def repetition_count(board: chess.Board) -> int:
    """Occurrences of the current position, saturated at five."""
    for count in range(5, 1, -1):
        if board.is_repetition(count):
            return count
    return 1
