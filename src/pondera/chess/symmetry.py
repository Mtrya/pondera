"""The exact board symmetry of standard chess as token and move transforms.

Standard chess admits exactly one nontrivial rule-preserving transform: the
rank mirror with colors swapped (white's perspective becomes black's). A file
mirror is NOT exact whenever castling rights exist: castling is defined with
the king on the e-file, so a file-mirrored position (king on d/e-file mirrored
to d) cannot hold castling rights and its legal move set does not commute with
the transform. Only the exact group {identity, mirror} is offered here.

TOKEN_MIRROR is a pure id permutation over the unified vocabulary: square
atoms mirror their square and swap piece color, castling rights swap K<->k and
Q<->q, and the en-passant square is mirrored; side, counters, moves, values,
and delimiters stay fixed. reflect_uci and move_permutations give the matching
transform on moves. All three are involutions: applying them twice restores
the input.
"""

from functools import lru_cache

import numpy as np

from pondera.chess.domain import IDX_TO_SQUARE, SQUARE_TO_IDX
from pondera.chess.vocabulary import (
    BOARD_TOKENS,
    CASTLING,
    CHESS_VOCAB_SIZE,
    EN_PASSANT,
    HALFMOVE,
    IDX_TO_UCI_MOVE,
    UCI_MOVE_TO_IDX,
)

# Square index (a1=0..h8=63) XOR shift for the rank mirror.
_MIRROR_SHIFT = 56


def _token_mirror():
    """Mirror every unified id; ids outside the board and metadata stay fixed."""
    table = np.arange(CHESS_VOCAB_SIZE, dtype=np.int64)
    atoms = np.arange(BOARD_TOKENS)
    table[:BOARD_TOKENS] = ((atoms // 12) ^ _MIRROR_SHIFT) * 12 + (atoms % 12 + 6) % 12
    rights = np.arange(CASTLING, EN_PASSANT) - CASTLING
    table[CASTLING:EN_PASSANT] = CASTLING + 2 * ((rights // 2) ^ 2) + rights % 2
    ep = np.arange(EN_PASSANT, HALFMOVE) - EN_PASSANT
    table[EN_PASSANT:HALFMOVE] = EN_PASSANT + np.where(ep != 64, ep ^ _MIRROR_SHIFT, ep)
    return table


TOKEN_MIRROR = _token_mirror()


def reflect_uci(move: str) -> str:
    """Rank-mirror a UCI move, the move transform of the black-to-move mirror.

    Move geometry commutes with the board mirror, so a legal move of a
    black-to-move root maps to a legal move of its mirrored board; the draw
    claim pseudo-move is not a board move and stays fixed.
    """
    if move == "<claim_draw>":
        return move
    return (
        IDX_TO_SQUARE[SQUARE_TO_IDX[move[:2]] ^ _MIRROR_SHIFT]
        + IDX_TO_SQUARE[SQUARE_TO_IDX[move[2:4]] ^ _MIRROR_SHIFT]
        + move[4:]
    )


@lru_cache(maxsize=1)
def move_permutations():
    """Map every move-vocabulary index through the mirror transform."""
    size = len(IDX_TO_UCI_MOVE)
    tables = np.empty((2, size), dtype=np.int64)
    tables[0] = np.arange(size)
    for index in range(size):
        tables[1, index] = UCI_MOVE_TO_IDX[reflect_uci(IDX_TO_UCI_MOVE[index])]
    return tables
