"""Synthetic trace datasets for encoder and training tests."""

from pathlib import Path

import chess
import numpy as np

from pondera.chess.codec import encode_board
from pondera.chess.vocabulary import CHESS_TOKENIZER_VERSION
from pondera.data.dataset import (
    DATA_FORMAT_VERSION,
    STATE_ARRAYS,
    write_manifest,
    write_trace_rows,
)
from pondera.data.trace import TRACE_GRAMMAR_VERSION


def play(moves, repetitions=None, start=chess.STARTING_FEN):
    """Records of (fen, move, repetition) for one game line."""
    board = chess.Board(start)
    repetitions = repetitions or [1] * len(moves)
    records = []
    for move, repetition in zip(moves, repetitions):
        records.append((board.fen(), move, repetition))
        board.push_uci(move)
    return records


def encoded_row(fen, tokens, roles, repetition=1):
    """One encoded trace row for write_trace_dataset."""
    return {
        "fen": fen,
        "repetition": repetition,
        "trace": np.asarray(tokens, dtype=np.int16),
        "roles": np.asarray(roles, dtype=np.int8),
    }


def write_trace_dataset(root, splits):
    """Write a trace dataset directory mirroring the published layout."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": DATA_FORMAT_VERSION,
        "grammar_version": TRACE_GRAMMAR_VERSION,
        "tokenizer_version": CHESS_TOKENIZER_VERSION,
    }
    manifest["shards"] = []
    for index, (split, rows) in enumerate(splits.items()):
        records = []
        for row in rows:
            records.append(
                {
                    "split": split,
                    "fen": row["fen"],
                    "repetition": row.get("repetition", 1),
                    "board_tokens": encode_board(row["fen"], row.get("repetition", 1)),
                    "trace_tokens": row["trace"],
                    "trace_len": len(row["trace"]),
                    "trace_roles": row["roles"],
                    **{name: [] for name in STATE_ARRAYS},
                }
            )
        name = f"shard-{index:06d}"
        write_trace_rows(root / name, records)
        manifest["shards"].append({"path": name, "rows": len(records)})
    write_manifest(root / "release.json", manifest)
    return root
