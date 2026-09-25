"""Lossless FEN tokens, with arbitrary-size counters encoded as byte digits.

The first 64 entries are squares a1 through h8. Typed metadata follows them:
side, four castling flags, en passant, and three delimited integer fields.
This format represents FEN plus a supplied repetition count, not game history.
"""

from collections.abc import Sequence

import numpy as np

from pondera.chess.mapping import (
    EMPTY_SQ_IDX,
    IDX_TO_PIECE,
    IDX_TO_SQUARE,
    PIECE_TO_IDX,
    SQUARE_TO_IDX,
)

BOARD_TOKENIZER_VERSION = 2
SIDE = 13
CASTLING = 15
EN_PASSANT = 23
COUNTER = 88
BYTE = 91
END = 347
PAD = 348
VOCAB_SIZE = 349
MIN_TOKENS = 79


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
