"""Encode a frozen annotation release into policy examples and legal action sets."""

import json
from collections import Counter
from hashlib import file_digest
from pathlib import Path

import chess
import numpy as np

from pondera.chess.board_tokens import PAD, encode_board
from pondera.chess.mapping import UCI_MOVE_TO_IDX
from pondera.chess.position import legal_move_indices
from pondera.data.storage import load_manifest, write_manifest


def _checksum(path):
    with Path(path).open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def prepare_teacher_policy(release_path, output_dir, depth=16):
    """Verify source hashes and reject split overlap before publishing encoded data.

    Terminal and incomplete-depth records are counted and excluded. Duplicate
    board positions within a split retain their first occurrence. Position keys
    use the first four FEN fields, preventing counter-only train/validation leaks.
    """
    release_path, output_dir = Path(release_path), Path(output_dir)
    release = load_manifest(release_path)
    if release.get("format_version") != 2 or set(release["files"]) != {"train", "val"}:
        raise ValueError("Expected a version-2 train/val annotation release")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1,
        "tokenizer_version": 2,
        "teacher_depth": depth,
        "release_sha256": _checksum(release_path),
        "position_key": "first four FEN fields",
        "legal_actions": "board moves; draw adjudication is external",
        "excluded": {},
    }
    validation_keys = set()
    teacher = None
    for split in ["val", "train"]:
        tokens, moves, legal_rows, seen = [], [], [], set()
        excluded = Counter()
        for shard in release["files"][split]:
            path = Path(shard["path"])
            if _checksum(path) != shard["sha256"]:
                raise ValueError(f"Source checksum mismatch: {path}")
            manifest = load_manifest(path.with_suffix(".manifest.json"))
            if manifest.get("format_version") != 2:
                raise ValueError("Annotation manifests must use root-relative format 2")
            if teacher is None:
                teacher = manifest
            elif teacher != manifest:
                raise ValueError("Teacher configuration changed within the release")
            count = 0
            with path.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    count += 1
                    key = " ".join(row["fen"].split()[:4])
                    if split == "train" and key in validation_keys:
                        raise ValueError(f"Training/validation position overlap: {key}")
                    if key in seen:
                        excluded["duplicate_position"] += 1
                        continue
                    seen.add(key)
                    if row.get("terminal"):
                        excluded["terminal"] += 1
                        continue
                    snapshot = next(
                        (
                            entry
                            for entry in row["unit"]["per_depth"]
                            if entry["d"] == depth
                        ),
                        None,
                    )
                    if snapshot is None:
                        excluded["missing_depth"] += 1
                        continue
                    board = chess.Board(row["fen"])
                    legal = legal_move_indices(board)
                    move = UCI_MOVE_TO_IDX[snapshot["r1"]["bm"]]
                    if move not in legal:
                        raise ValueError(
                            "Teacher move is illegal in its source position"
                        )
                    tokens.append(encode_board(row["fen"], row.get("repetition", 1)))
                    moves.append(move)
                    legal_rows.append(legal)
            if count != shard["rows"]:
                raise ValueError(f"Source row count mismatch: {path}")
        if not tokens:
            if release["files"][split]:
                raise ValueError(f"No eligible examples in {split}")
            metadata[f"rows_{split}"] = 0
            metadata["excluded"][split] = dict(excluded)
            continue
        if split == "val":
            validation_keys = seen
        for name, rows, padding in [("tokens", tokens, PAD), ("legal", legal_rows, -1)]:
            array = np.lib.format.open_memmap(
                output_dir / f"{name}_{split}.npy",
                mode="w+",
                dtype=np.int16,
                shape=(len(rows), max(map(len, rows))),
            )
            array[:] = padding
            for index, values in enumerate(rows):
                array[index, : len(values)] = values
            array.flush()
        np.save(output_dir / f"moves_{split}.npy", np.asarray(moves, dtype=np.int16))
        metadata[f"rows_{split}"] = len(tokens)
        metadata["excluded"][split] = dict(excluded)
    metadata["teacher"] = teacher
    metadata["files"] = {
        path.name: _checksum(path) for path in sorted(output_dir.glob("*.npy"))
    }
    write_manifest(output_dir / "manifest.json", metadata)
    return metadata
