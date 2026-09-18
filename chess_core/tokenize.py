"""Fast FEN -> token-id encoder for offline precomputation and runtime use.

Token layout (73 ids per position, uint8):
    0-63:  piece id per square (a1=0 ... h8=63), values 0-12 (PIECE_TO_IDX)
    64:    side to move (0=white, 1=black)
    65-68: castling rights K, Q, k, q (1=present, 0=absent)
    69:    en passant target square index (0-63) or EP_NONE(64) for '-'
    70:    halfmove clock, clamped to [0, MAX_HALFMOVES-1]
    71:    fullmove number, clamped to [1, MAX_FULLMOVES] then 0-based
    72:    repetition count, clamped to [1,3] then 0-based

The encoding replicates model.FENTokenizer's string parsing exactly; run
`python -m chess_core.tokenize` for a parity check against the string path.
"""

import multiprocessing
from typing import List, Optional, Sequence

import numpy as np

from .mapping import EMPTY_SQ_IDX, MAX_FULLMOVES, MAX_HALFMOVES, PIECE_TO_IDX, SQUARE_TO_IDX

TOKEN_COUNT = 73
EP_NONE = 64
TOKENIZER_VERSION = 1


def encode_fen(fen: str, repetition: int = 1) -> np.ndarray:
    """Encode a single FEN (+ repetition count) into 73 uint8 token ids."""
    parts = fen.split()
    if len(parts) != 6:
        raise ValueError(f"Invalid FEN string: {fen}. Expected 6 fields")
    placement, side, castling, en_passant, halfmove, fullmove = parts

    out = np.empty(TOKEN_COUNT, dtype=np.uint8)

    squares = np.full(64, EMPTY_SQ_IDX, dtype=np.uint8)
    rank, file = 7, 0
    for ch in placement:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            squares[rank * 8 + file] = PIECE_TO_IDX[ch]
            file += 1
    out[0:64] = squares

    out[64] = 0 if side == "w" else 1

    out[65] = 1 if "K" in castling else 0
    out[66] = 1 if "Q" in castling else 0
    out[67] = 1 if "k" in castling else 0
    out[68] = 1 if "q" in castling else 0

    out[69] = EP_NONE if en_passant == "-" else SQUARE_TO_IDX[en_passant]

    out[70] = min(max(int(halfmove), 0), MAX_HALFMOVES - 1)
    out[71] = min(max(int(fullmove), 1), MAX_FULLMOVES) - 1
    out[72] = min(max(repetition - 1, 0), 2)
    return out


def _encode_worker(args) -> np.ndarray:
    fen, rep = args
    return encode_fen(fen, rep)


def encode_fens(
    fens: Sequence[str],
    repetitions: Optional[Sequence[int]] = None,
    num_workers: int = 0,
) -> np.ndarray:
    """Batch-encode FENs into an (N, 73) uint8 array.

    num_workers=0 (default) runs sequentially; >0 uses a multiprocessing pool.
    """
    n = len(fens)
    if repetitions is None:
        repetitions = [1] * n
    if num_workers and num_workers > 1 and n > 1:
        with multiprocessing.Pool(processes=num_workers) as pool:
            rows = pool.map(_encode_worker, zip(fens, repetitions), chunksize=1024)
        return np.stack(rows)
    out = np.empty((n, TOKEN_COUNT), dtype=np.uint8)
    for i, (fen, rep) in enumerate(zip(fens, repetitions)):
        out[i] = encode_fen(fen, rep)
    return out


def _parity_check(num_positions: int = 512, hidden_size: int = 16) -> None:
    """Verify forward_ids matches FENTokenizer's string path on real positions."""
    import random

    import chess
    import torch

    from model import FENTokenizer

    random.seed(0)
    fens, reps = [], []
    for _ in range(num_positions):
        board = chess.Board()
        for _ply in range(random.randint(0, 120)):
            if board.is_game_over():
                break
            board.push(random.choice(list(board.legal_moves)))
        fens.append(board.fen())
        reps.append(random.randint(1, 3))

    tokenizer = FENTokenizer(hidden_size, dtype=None)
    ids = torch.from_numpy(encode_fens(fens, reps).astype(np.int64))
    ref = tokenizer(fens, torch.tensor(reps))
    got = tokenizer.forward_ids(ids)
    assert torch.allclose(ref, got), "parity check failed"
    print(f"parity check passed on {num_positions} positions")


if __name__ == "__main__":
    _parity_check()
