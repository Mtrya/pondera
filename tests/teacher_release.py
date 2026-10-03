"""Synthetic annotation releases for encoder tests."""

import json
from hashlib import file_digest
from pathlib import Path

import chess
import numpy as np

from pondera.chess.tokens import PAD, encode_board
from pondera.data.storage import write_manifest
from pondera.data.trace import PV_LIMIT, depth_snapshot, encode_trace, max_segments


def play(moves, repetitions=None, start=chess.STARTING_FEN):
    """Records of (fen, move, repetition) for one game line."""
    board = chess.Board(start)
    repetitions = repetitions or [1] * len(moves)
    records = []
    for move, repetition in zip(moves, repetitions):
        records.append((board.fen(), move, repetition))
        board.push_uci(move)
    return records


def write_teacher_release(root, splits, depth=1):
    """Write a minimal version-2 annotation release and return its manifest path.

    Positions must be unique within a split and disjoint across splits, matching
    the integrity checks of the teacher encoder.
    """
    root = Path(root)
    release = {"format_version": 2, "files": {}}
    for split, records in splits.items():
        if not records:
            release["files"][split] = []
            continue
        path = root / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for fen, move, repetition in records:
                handle.write(
                    json.dumps(
                        {
                            "fen": fen,
                            "repetition": repetition,
                            "unit": {"per_depth": [{"d": depth, "r1": {"bm": move}}]},
                        }
                    )
                    + "\n"
                )
        write_manifest(path.with_suffix(".manifest.json"), {"format_version": 2})
        with path.open("rb") as handle:
            digest = file_digest(handle, "sha256").hexdigest()
        release["files"][split] = [
            {"path": str(path), "sha256": digest, "rows": len(records)}
        ]
    manifest = root / "release.json"
    write_manifest(manifest, release)
    return manifest


def depth_row(fen, ranks):
    """One release row with per-depth rank-1 labels: (depth, move, mate, value)."""
    return {
        "fen": fen,
        "repetition": 1,
        "unit": {
            "per_depth": [
                {"d": depth, "r1": {"bm": move, "mate": mate, "value": value}}
                for depth, move, mate, value in ranks
            ]
        },
    }


def write_depth_release(root, splits):
    """Write a version-2 release whose rows carry complete per-depth labels."""
    root = Path(root)
    release = {"format_version": 2, "files": {}}
    for split, rows in splits.items():
        path = root / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        write_manifest(path.with_suffix(".manifest.json"), {"format_version": 2})
        with path.open("rb") as handle:
            digest = file_digest(handle, "sha256").hexdigest()
        release["files"][split] = [
            {"path": str(path), "sha256": digest, "rows": len(rows)}
        ]
    manifest = root / "release.json"
    write_manifest(manifest, release)
    return manifest


def candidate_row(fen, candidates, depth=16, repetition=1):
    """One release row whose depth snapshot carries ranked candidates.

    candidates is a list of (bm, pv, mate, value) in teacher rank order.
    """
    snapshot = {"d": depth}
    for index, (bm, pv, mate, value) in enumerate(candidates, 1):
        snapshot[f"r{index}"] = {
            "bm": bm,
            "pv": list(pv),
            "mate": mate,
            "value": value,
        }
    return {
        "fen": fen,
        "repetition": repetition,
        "unit": {"per_depth": [snapshot]},
    }


def candidate_entries(board, count=3, plies=12):
    """(bm, pv, mate, value) teacher candidates with replayed first-legal PVs."""
    entries = []
    for index, move in enumerate(
        sorted(board.legal_moves, key=lambda m: m.uci())[:count]
    ):
        pv, replay, current = [], board.copy(), move
        for _ in range(plies):
            pv.append(current.uci())
            replay.push(current)
            legal = sorted(replay.legal_moves, key=lambda m: m.uci())
            if not legal:
                break
            current = legal[0]
        entries.append((pv[0], pv, 3 if index == 0 else None, 0.5))
    return entries


def write_sequence_dataset(root, splits, level="level15", depth=16):
    """Write a teacher-sequence dataset directory mirroring the published layout."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "kind": "teacher-sequence",
        "level": level,
        "tokenizer_version": 2,
        "teacher_depth": depth,
        "pv_limit": PV_LIMIT,
        "max_lines": max_segments(level),
    }
    for split, rows in splits.items():
        boards = [encode_board(row["fen"], row.get("repetition", 1)) for row in rows]
        width = max(len(board) for board in boards)
        board_array = np.full((len(rows), width), PAD, dtype=np.int16)
        for index, board in enumerate(boards):
            board_array[index, : len(board)] = board
        encoded = [
            encode_trace(
                depth_snapshot(row, depth),
                row["fen"].split()[1] == "b",
                level,
                board=chess.Board(row["fen"]),
            )
            for row in rows
        ]
        trace_width = max(len(tokens) for tokens, _ in encoded)
        trace_array = np.full((len(rows), trace_width), PAD, dtype=np.int16)
        length_array = np.zeros(len(rows), dtype=np.int16)
        bound_array = np.full((len(rows), max_segments(level), 2), -1, dtype=np.int16)
        for index, (tokens, bounds) in enumerate(encoded):
            trace_array[index, : len(tokens)] = tokens
            length_array[index] = len(tokens)
            for slot, (start, end) in enumerate(bounds):
                bound_array[index, slot] = (start, end)
        for name, array in (
            ("board_tokens", board_array),
            ("trace_tokens", trace_array),
            ("trace_len", length_array),
            ("line_bounds", bound_array),
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
