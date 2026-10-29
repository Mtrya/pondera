"""Conversions between chess states and the unified vocabulary ids.

A board encoding is the set of occupied squares as joint (square, piece) atoms
in a1..h8 scan order, followed by eight metadata atoms: side to move, four
castling rights, en passant, halfmove clock, and repetition count. Every atom
is one complete statement; an empty square is the absence of an atom. The
encoding is lossless for the first five FEN fields. The fullmove counter is
not encoded, so decoding restores the FEN modulo that counter; repetition
counts above four share the top atom.

The id layout itself lives in `pondera.chess.vocabulary`; this module only
converts. Value-bin boundaries and representatives are the numeric semantics
of the value segment, frozen on the pinned teacher release.
"""

import bisect
from collections.abc import Sequence

import chess
import numpy as np

from pondera.chess.domain import (
    EMPTY_SQ_IDX,
    IDX_TO_PIECE,
    IDX_TO_SQUARE,
    PIECE_TO_IDX,
    SQUARE_TO_IDX,
)
from pondera.chess.vocabulary import (
    BOARD_PIECES,
    CASTLING,
    EN_PASSANT,
    HALFMOVE,
    MATE_LOSS,
    MATE_WIN,
    META_END,
    PAD,
    REPETITION,
    SIDE,
    UCI_MOVE_TO_IDX,
    VALUE,
)

# Dataset guardrail: one encoded position has at most 32 piece atoms and 8
# metadata atoms; the guard allows headroom above that bound.
BOARD_TOKENS_MAX = 42

# Value bins, frozen on the pinned teacher release (pooled candidate ranks,
# non-mate WDL-derived values in [-1, 1]): bin 0 and bin 19 are the -1 and +1
# atoms; the interior bins are equiprobable with (lo, hi] membership.
VALUE_BIN_BOUNDARIES = (
    -0.9995,
    -0.9935,
    -0.9495,
    -0.7525,
    -0.4055,
    -0.1665,
    -0.0685,
    -0.0265,
    -0.0065,
    -0.0005,
    0.0045,
    0.0185,
    0.0485,
    0.1125,
    0.2795,
    0.6245,
    0.9145,
    0.9915,
    0.9995,
)
# Mass-weighted mean of each bin on the same frozen fit: the MSE-optimal decode.
VALUE_BIN_REPRESENTATIVES = (
    -1.0,
    -0.9975466656545338,
    -0.9778050997401598,
    -0.869374566711709,
    -0.5802287566316574,
    -0.27132291636100436,
    -0.11024468967970123,
    -0.0449997169102215,
    -0.014999041330398631,
    -0.0031977730105220144,
    0.000901943418371388,
    0.010669587162031527,
    0.03184649072259998,
    0.07566960512966994,
    0.18261132375456368,
    0.4405164642839585,
    0.7900643834679745,
    0.9648939514177877,
    0.997022958525894,
    1.0,
)
# Frozen mate coverage of the pinned teacher data; distance 0 is reserved for
# the terminal position after perspective normalization.
MATE_WIN_MAX = 28
MATE_LOSS_MAX = 30


def value_bin(value: float) -> int:
    """Assign a WDL-derived scalar in [-1, 1] to its frozen bin."""
    if not -1.0 <= value <= 1.0:
        raise ValueError("Teacher values lie in [-1, 1]")
    return bisect.bisect_left(VALUE_BIN_BOUNDARIES, value)


def mate_token(distance: int, win: bool) -> int:
    """Token id for a mate announcement from the moving side's perspective."""
    if not 0 <= distance <= (MATE_WIN_MAX if win else MATE_LOSS_MAX):
        raise ValueError("Mate distance outside the frozen coverage")
    return (MATE_WIN if win else MATE_LOSS) + distance


def decode_value_token(token: int) -> float:
    """Decode a value or mate token to a scalar; mate distance decodes to +-1."""
    if VALUE <= token < VALUE + len(VALUE_BIN_REPRESENTATIVES):
        return VALUE_BIN_REPRESENTATIVES[token - VALUE]
    if MATE_WIN <= token <= MATE_WIN + MATE_WIN_MAX:
        return 1.0
    if MATE_LOSS <= token <= MATE_LOSS + MATE_LOSS_MAX:
        return -1.0
    raise ValueError(f"Not a value token: {token}")


def legal_move_indices(board: chess.Board) -> list[int]:
    """Board moves only; draw adjudication belongs to the game runner."""
    return sorted(UCI_MOVE_TO_IDX[move.uci()] for move in board.legal_moves)


def encode_board(fen: str, repetition: int = 1) -> np.ndarray:
    """Encode a FEN as occupied-square atoms plus self-contained metadata.

    The fullmove counter is not part of the vocabulary; callers keep the
    source FEN when it matters. Halfmove clocks outside 0..100 and repetition
    counts above five cannot be represented and are rejected or shared with
    the top repetition atom respectively.
    """
    placement, side, castling, ep, halfmove, fullmove = fen.split()
    rows = placement.split("/")
    if len(rows) != 8 or side not in {"w", "b"}:
        raise ValueError("FEN needs eight ranks and a valid side to move")
    squares = []
    for row in reversed(rows):
        rank = []
        for char in row:
            if char in "12345678":
                rank.extend([EMPTY_SQ_IDX] * int(char))
            elif char in PIECE_TO_IDX and char != ".":
                rank.append(PIECE_TO_IDX[char])
            else:
                raise ValueError(f"Invalid placement character: {char}")
        if len(rank) != 8:
            raise ValueError("Each FEN rank must contain eight squares")
        squares.extend(rank)
    if castling != "-" and (
        not castling
        or set(castling) - set("KQkq")
        or len(set(castling)) != len(castling)
    ):
        raise ValueError("Expected standard-chess castling rights")
    if ep != "-" and ep not in SQUARE_TO_IDX:
        raise ValueError("Invalid en-passant square")
    halfmove_count, fullmove_count = int(halfmove), int(fullmove)
    if not 0 <= halfmove_count <= 100:
        raise ValueError("Halfmove clock must lie in 0..100")
    if fullmove_count < 1 or repetition < 1:
        raise ValueError("Invalid fullmove or repetition count")
    tokens = [
        square * BOARD_PIECES + piece
        for square, piece in enumerate(squares)
        if piece != EMPTY_SQ_IDX
    ]
    tokens.append(SIDE + (side == "b"))
    tokens.extend(
        CASTLING + 2 * index + (right in castling) for index, right in enumerate("KQkq")
    )
    tokens.append(EN_PASSANT + (64 if ep == "-" else SQUARE_TO_IDX[ep]))
    tokens.append(HALFMOVE + halfmove_count)
    tokens.append(REPETITION + min(repetition, 5) - 1)
    return np.asarray(tokens, dtype=np.int16)


def decode_board(tokens: Sequence[int]) -> tuple[str, int]:
    """Recover the FEN (fullmove fixed to 1) and the repetition atom.

    Board atoms are order-independent; duplicate square atoms and malformed
    metadata raise. Metadata atoms are identified by their own ids, so a
    mirrored row, whose castling atoms move between fields, still decodes.
    Counts above four decode as the shared top atom, five.
    """
    values = [int(value) for value in tokens]
    while values and values[-1] == PAD:
        values.pop()
    squares = [EMPTY_SQ_IDX] * 64
    index = 0
    while index < len(values) and values[index] < SIDE:
        square, piece = divmod(values[index], BOARD_PIECES)
        if squares[square] != EMPTY_SQ_IDX:
            raise ValueError("Duplicate board square atom")
        squares[square] = piece
        index += 1
    if not any(piece != EMPTY_SQ_IDX for piece in squares):
        raise ValueError("A board encoding needs at least one piece atom")
    metadata = values[index:]
    if len(metadata) != 8:
        raise ValueError("A board encoding needs eight metadata atoms")
    side = [token for token in metadata if token in (SIDE, SIDE + 1)]
    castling = [token for token in metadata if CASTLING <= token < EN_PASSANT]
    ep = [token for token in metadata if EN_PASSANT <= token < HALFMOVE]
    halfmove = [token for token in metadata if HALFMOVE <= token < REPETITION]
    repetition = [token for token in metadata if REPETITION <= token <= META_END]
    if len(side) != 1:
        raise ValueError("Invalid side atom")
    flags = {}
    for token in castling:
        field, flag = divmod(token - CASTLING, 2)
        if field in flags:
            raise ValueError("Duplicate castling atom")
        flags[field] = flag
    if len(flags) != 4:
        raise ValueError("Missing castling atom")
    present = ["KQkq"[field] for field in range(4) if flags[field]]
    if len(ep) != 1 or len(halfmove) != 1 or len(repetition) != 1:
        raise ValueError("Missing or repeated metadata atom")
    ep_square = ep[0] - EN_PASSANT
    halfmove_count = halfmove[0] - HALFMOVE
    repetition_count = repetition[0] - REPETITION
    if not 0 <= ep_square <= 64:
        raise ValueError("Invalid en-passant atom")
    if not 0 <= halfmove_count <= 100:
        raise ValueError("Invalid halfmove atom")
    if not 0 <= repetition_count < 5:
        raise ValueError("Invalid repetition atom")
    rows = []
    for rank in range(7, -1, -1):
        text, empty = "", 0
        for piece in squares[rank * 8 : (rank + 1) * 8]:
            if piece == EMPTY_SQ_IDX:
                empty += 1
            else:
                text += (str(empty) if empty else "") + IDX_TO_PIECE[piece]
                empty = 0
        rows.append(text + (str(empty) if empty else ""))
    fen = " ".join(
        [
            "/".join(rows),
            "w" if side[0] == SIDE else "b",
            "".join(present) or "-",
            "-" if ep_square == 64 else IDX_TO_SQUARE[ep_square],
            str(halfmove_count),
            "1",
        ]
    )
    return fen, repetition_count + 1


def encode_boards(fens, repetitions=None) -> np.ndarray:
    """Pad a batch to its longest encoded position."""
    if repetitions is None:
        repetitions = [1] * len(fens)
    if len(fens) != len(repetitions):
        raise ValueError("Each FEN needs a repetition count")
    rows = [encode_board(fen, rep) for fen, rep in zip(fens, repetitions)]
    width = max((len(row) for row in rows), default=0)
    result = np.full((len(rows), width), PAD, dtype=np.int16)
    for i, row in enumerate(rows):
        result[i, : len(row)] = row
    return result
