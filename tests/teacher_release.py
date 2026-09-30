"""Synthetic annotation releases for encoder tests."""

import json
from hashlib import file_digest
from pathlib import Path

import chess

from pondera.data.storage import write_manifest


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
