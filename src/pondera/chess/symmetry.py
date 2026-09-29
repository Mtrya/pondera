"""The exact board symmetry of standard chess as a batched data transform.

Standard chess admits exactly one nontrivial rule-preserving transform: the
rank mirror with colors swapped (white's perspective becomes black's). A file
mirror is NOT exact whenever castling rights exist: castling is defined with
the king on the e-file, so a file-mirrored position (king on d/e-file mirrored
to d) cannot hold castling rights and its legal move set does not commute with
the transform. Only the exact group {identity, mirror} is offered here. Each
transform acts on lossless token rows (unpacked, as stored in the frozen
datasets) and on move vocabulary indices, so a (position, label, legal) triple
can be re-oriented without re-encoding from FEN. Both transforms are
involutions: applying them twice restores the input exactly.
"""

from functools import lru_cache

import numpy as np

from pondera.chess.tokens import CASTLING, EN_PASSANT, SIDE
from pondera.chess.mapping import (
    EMPTY_SQ_IDX,
    IDX_TO_SQUARE,
    IDX_TO_UCI_MOVE,
    SQUARE_TO_IDX,
    UCI_MOVE_TO_IDX,
)

NUM_TRANSFORMS = 2

# Square index (a1=0..h8=63) XOR shift for the rank mirror.
_MIRROR_SHIFT = 56

# Castling fields are ordered K, Q, k, q; the mirror moves the right stored in
# field j ^ 2 into field j (K<->k, Q<->q).
_CASTLING_SOURCE = (2, 3, 0, 1)

MIRROR_PERMUTATION = np.arange(64) ^ _MIRROR_SHIFT


@lru_cache(maxsize=1)
def move_permutations():
    """Map every move-vocabulary index through each transform."""
    size = len(IDX_TO_UCI_MOVE)
    tables = np.empty((NUM_TRANSFORMS, size), dtype=np.int64)
    tables[0] = np.arange(size)
    for index in range(size):
        uci = IDX_TO_UCI_MOVE[index]
        if uci == "<claim_draw>":
            tables[1, index] = index
            continue
        mapped = (
            IDX_TO_SQUARE[SQUARE_TO_IDX[uci[:2]] ^ _MIRROR_SHIFT]
            + IDX_TO_SQUARE[SQUARE_TO_IDX[uci[2:4]] ^ _MIRROR_SHIFT]
            + uci[4:]
        )
        tables[1, index] = UCI_MOVE_TO_IDX[mapped]
    return tables


def apply_to_tokens(ids, transforms):
    """Re-orient unpacked lossless token rows; counters and padding stay fixed."""
    ids = np.asarray(ids)
    transforms = np.asarray(transforms)
    if ids.ndim != 2 or transforms.shape != (ids.shape[0],):
        raise ValueError("Expected a row batch with one transform index per row")
    out = ids.copy()
    mask = transforms == 1
    if not mask.any():
        return out
    rows = ids[mask]
    new = rows.copy()
    new[:, :64] = rows[:, MIRROR_PERMUTATION]
    board = new[:, :64]
    occupied = board < EMPTY_SQ_IDX
    board[occupied] = (board[occupied] + 6) % 12
    new[:, 64] = 2 * SIDE + 1 - rows[:, 64]
    castling_fields = CASTLING + 2 * np.arange(4)
    flags = rows[:, 65:69] - castling_fields
    new[:, 65:69] = castling_fields + flags[:, _CASTLING_SOURCE]
    ep = rows[:, 69] - EN_PASSANT
    new[:, 69] = EN_PASSANT + np.where(ep != 64, ep ^ _MIRROR_SHIFT, 64)
    out[mask] = new
    return out


def apply_to_moves(moves, transforms):
    """Map move-vocabulary indices through the per-row transforms."""
    moves = np.asarray(moves)
    return move_permutations()[np.asarray(transforms), moves]


def apply_to_legal(legal, transforms):
    """Map legal-move arrays, preserving the -1 fill value."""
    legal = np.asarray(legal)
    transformed = move_permutations()[
        np.asarray(transforms)[:, None], np.clip(legal, 0, None)
    ]
    return np.where(legal >= 0, transformed, -1).astype(legal.dtype)
