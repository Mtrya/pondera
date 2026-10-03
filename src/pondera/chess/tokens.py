"""Lossless FEN tokens and the unified chess token vocabulary.

The first 64 entries of a board encoding are squares a1 through h8. Typed
metadata follows them: side, four castling flags, en passant, and three
delimited integer fields. This format represents FEN plus a supplied
repetition count, not game history.

CHESS_TOKENIZER extends the board format into one shared id space: board
tokens keep their historical ids, followed by blocks for the move vocabulary,
value bins, and mate distances. Sequence models consume and emit this unified
space; the fixed-depth board policy keeps its dedicated action head.
"""

import bisect
from collections.abc import Sequence

import numpy as np

from pondera.chess.mapping import (
    EMPTY_SQ_IDX,
    IDX_TO_PIECE,
    IDX_TO_SQUARE,
    IDX_TO_UCI_MOVE,
    PIECE_TO_IDX,
    SQUARE_TO_IDX,
)

# The version pins the serialized board-token format recorded in data
# manifests and checkpoints; extending the unified id space does not change it.
CHESS_TOKENIZER_VERSION = 2
SIDE = 13
CASTLING = 15
EN_PASSANT = 23
COUNTER = 88
BYTE = 91
END = 347
PAD = 348
VOCAB_SIZE = 349
MIN_TOKENS = 79

# Unified id space. Board tokens occupy 0..VOCAB_SIZE-1; later blocks follow
# without gaps and extend append-only.
MOVE = VOCAB_SIZE
VALUE = MOVE + len(IDX_TO_UCI_MOVE)
MATE_WIN = VALUE + 20
MATE_LOSS = MATE_WIN + 29
# Sequence trace delimiters: LINE opens a candidate line, FINAL opens the
# answer; END closes both a board counter field and a trace.
LINE = MATE_LOSS + 31
FINAL = LINE + 1
# Explicit board snapshots inside a trace: joint (square, piece) tokens over
# the twelve pieces of the board encoding, square-major in a1..h8 scan order.
# Empty squares carry no token; the metadata block that follows reuses the
# board-token side, castling, and en-passant ids.
SNAPSHOT = FINAL + 1
SNAPSHOT_PIECES = 12
CHESS_VOCAB_SIZE = SNAPSHOT + SNAPSHOT_PIECES * 64

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


def snapshot_token(square: int, piece: int) -> int:
    """Joint token id of one occupied square in a board snapshot."""
    if not 0 <= square < 64:
        raise ValueError("Snapshot squares lie in 0..63")
    if not 0 <= piece < SNAPSHOT_PIECES:
        raise ValueError("Snapshot pieces exclude the empty square")
    return SNAPSHOT + square * SNAPSHOT_PIECES + piece


def decode_snapshot_token(token: int) -> tuple[int, int]:
    """Recover the (square, piece) pair of a snapshot token id."""
    if not SNAPSHOT <= token < CHESS_VOCAB_SIZE:
        raise ValueError(f"Not a snapshot token: {token}")
    return divmod(token - SNAPSHOT, SNAPSHOT_PIECES)


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


def encode_board(fen: str, repetition: int = 1) -> np.ndarray:
    """Encode the semantic contents of a standard-chess FEN without clipping."""
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
    counts = (int(halfmove), int(fullmove), int(repetition))
    if counts[0] < 0 or counts[1] < 1 or counts[2] < 1 or repetition != counts[2]:
        raise ValueError("Invalid halfmove, fullmove, or repetition count")
    tokens = squares + [SIDE + (side == "b")]
    tokens.extend(
        CASTLING + 2 * i + (right in castling) for i, right in enumerate("KQkq")
    )
    tokens.append(EN_PASSANT + (64 if ep == "-" else SQUARE_TO_IDX[ep]))
    for field, value in enumerate(counts):
        tokens.append(COUNTER + field)
        digits = value.to_bytes(max(1, (value.bit_length() + 7) // 8), "little")
        tokens.extend(BYTE + digit for digit in digits)
        tokens.append(END)
    return np.asarray(tokens, dtype=np.int16)


def decode_board(tokens: Sequence[int]) -> tuple[str, int]:
    """Recover a canonical FEN and exact repetition count from encoded tokens."""
    values = [int(value) for value in tokens]
    while values and values[-1] == PAD:
        values.pop()
    if len(values) < MIN_TOKENS or any(not 0 <= v <= 12 for v in values[:64]):
        raise ValueError("Invalid board token sequence")
    rows = []
    for rank in range(7, -1, -1):
        text, empty = "", 0
        for piece in values[rank * 8 : (rank + 1) * 8]:
            if piece == EMPTY_SQ_IDX:
                empty += 1
            else:
                text += (str(empty) if empty else "") + IDX_TO_PIECE[piece]
                empty = 0
        rows.append(text + (str(empty) if empty else ""))
    if values[64] not in (SIDE, SIDE + 1):
        raise ValueError("Invalid side token")
    rights = ""
    for i, right in enumerate("KQkq"):
        flag = values[65 + i] - CASTLING - 2 * i
        if flag not in (0, 1):
            raise ValueError("Invalid castling token")
        if flag:
            rights += right
    ep = values[69] - EN_PASSANT
    if not 0 <= ep <= 64:
        raise ValueError("Invalid en-passant token")
    cursor, counts = 70, []
    for field in range(3):
        if cursor >= len(values) or values[cursor] != COUNTER + field:
            raise ValueError("Missing counter field")
        cursor += 1
        digits = []
        while cursor < len(values) and BYTE <= values[cursor] < BYTE + 256:
            digits.append(values[cursor] - BYTE)
            cursor += 1
        if not digits or cursor >= len(values) or values[cursor] != END:
            raise ValueError("Invalid counter digits")
        counts.append(int.from_bytes(bytes(digits), "little"))
        cursor += 1
    if cursor != len(values) or counts[1] < 1 or counts[2] < 1:
        raise ValueError("Invalid trailing tokens or counter value")
    fen = " ".join(
        [
            "/".join(rows),
            "w" if values[64] == SIDE else "b",
            rights or "-",
            "-" if ep == 64 else IDX_TO_SQUARE[ep],
            str(counts[0]),
            str(counts[1]),
        ]
    )
    return fen, counts[2]


def encode_boards(fens, repetitions=None) -> np.ndarray:
    """Pad a batch to its longest lossless representation."""
    if repetitions is None:
        repetitions = [1] * len(fens)
    if len(fens) != len(repetitions):
        raise ValueError("Each FEN needs a repetition count")
    rows = [encode_board(fen, rep) for fen, rep in zip(fens, repetitions)]
    width = max((len(row) for row in rows), default=MIN_TOKENS)
    result = np.full((len(rows), width), PAD, dtype=np.int16)
    for i, row in enumerate(rows):
        result[i, : len(row)] = row
    return result
