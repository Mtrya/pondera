"""Synthetic sequence datasets for encoder and training tests."""

from hashlib import file_digest
from pathlib import Path

import chess
import numpy as np

from pondera.chess.codec import encode_board
from pondera.chess.vocabulary import CHESS_TOKENIZER_VERSION, PAD
from pondera.data.sequence_format import SEQUENCE_FORMAT_VERSION
from pondera.data.storage import write_manifest
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


def sequence_row(fen, tokens, roles, repetition=1):
    """One encoded trace row for write_sequence_dataset."""
    return {
        "fen": fen,
        "repetition": repetition,
        "trace": np.asarray(tokens, dtype=np.int16),
        "roles": np.asarray(roles, dtype=np.int8),
    }


def write_sequence_dataset(root, splits, depth=16):
    """Write a sequence dataset directory mirroring the published layout."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": SEQUENCE_FORMAT_VERSION,
        "kind": "teacher-sequence",
        "grammar_version": TRACE_GRAMMAR_VERSION,
        "tokenizer_version": CHESS_TOKENIZER_VERSION,
        "depth": depth,
    }
    for split, rows in splits.items():
        boards = [encode_board(row["fen"], row.get("repetition", 1)) for row in rows]
        width = max(len(board) for board in boards)
        board_array = np.full((len(rows), width), PAD, dtype=np.int16)
        for index, board in enumerate(boards):
            board_array[index, : len(board)] = board
        trace_width = max(len(row["trace"]) for row in rows)
        trace_array = np.full((len(rows), trace_width), PAD, dtype=np.int16)
        length_array = np.zeros(len(rows), dtype=np.int32)
        role_array = np.full((len(rows), trace_width), 0, dtype=np.int8)
        for index, row in enumerate(rows):
            tokens, roles = row["trace"], row["roles"]
            trace_array[index, : len(tokens)] = tokens
            role_array[index, : len(roles)] = roles
            length_array[index] = len(tokens)
        for name, array in (
            ("board_tokens", board_array),
            ("trace_tokens", trace_array),
            ("trace_len", length_array),
            ("trace_roles", role_array),
        ):
            np.save(root / f"{name}_{split}.npy", array)
        manifest[f"rows_{split}"] = len(rows)
        manifest[f"t_max_{split}"] = trace_width
    manifest["files"] = {}
    for path in sorted(root.glob("*.npy")):
        with path.open("rb") as handle:
            manifest["files"][path.name] = file_digest(handle, "sha256").hexdigest()
    write_manifest(root / "manifest.json", manifest)
    return root
