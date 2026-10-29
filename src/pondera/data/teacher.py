"""Encode a frozen annotation release into policy examples and legal action sets."""

import json
from collections import Counter
from hashlib import file_digest
from pathlib import Path

import chess
import numpy as np

from pondera.chess.codec import (
    encode_board,
    legal_move_indices,
    mate_token,
    value_bin,
)
from pondera.chess.domain import EMPTY_SQ_IDX
from pondera.chess.vocabulary import (
    CHESS_TOKENIZER_VERSION,
    PAD,
    UCI_MOVE_TO_IDX,
    VALUE,
)
from pondera.data.storage import load_manifest, write_manifest


def _checksum(path):
    with Path(path).open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def _depth_targets(row, max_depth):
    """Per-depth rank-1 targets, or None when the row is not fully annotated."""
    snapshots = {entry["d"]: entry["r1"] for entry in row["unit"]["per_depth"]}
    if any(depth not in snapshots for depth in range(1, max_depth + 1)):
        return None
    moves, value_tokens, values = [], [], []
    for depth in range(1, max_depth + 1):
        rank = snapshots[depth]
        moves.append(UCI_MOVE_TO_IDX[rank["bm"]])
        if rank["mate"] is not None:
            value_tokens.append(mate_token(abs(rank["mate"]), rank["mate"] > 0))
        else:
            value_tokens.append(VALUE + value_bin(rank["value"]))
        values.append(rank["value"])
    return moves, value_tokens, values


def _after_board(current, board, move):
    """Absolute square-content classes after pushing move on the pre-push board."""
    after = current.copy()
    moving = after[move.from_square]
    after[move.from_square] = EMPTY_SQ_IDX
    if move.promotion:
        after[move.to_square] = (move.promotion - 1) + (0 if moving < 6 else 6)
    else:
        after[move.to_square] = moving
    if board.is_en_passant(move):
        after[move.to_square + (-8 if board.turn else 8)] = EMPTY_SQ_IDX
    if board.is_castling(move):
        if move.to_square > move.from_square:
            rook_from, rook_to = move.from_square + 3, move.from_square + 1
        else:
            rook_from, rook_to = move.from_square - 4, move.from_square - 1
        after[rook_to] = after[rook_from]
        after[rook_from] = EMPTY_SQ_IDX
    return after


def _canonical(array, black_to_move):
    """Square-content classes in side-to-move orientation (rank mirror + color swap)."""
    if not black_to_move:
        return array
    mirrored = array[np.arange(64) ^ 56]
    return np.where(mirrored < 12, (mirrored + 6) % 12, mirrored).astype(np.uint8)


def _board_aux(row, max_depth):
    """Canonical after-content and changed masks, one per teacher depth, or None."""
    snapshots = {entry["d"]: entry["r1"] for entry in row["unit"]["per_depth"]}
    if any(depth not in snapshots for depth in range(1, max_depth + 1)):
        return None
    board = chess.Board(row["fen"])
    current_absolute = np.full(64, EMPTY_SQ_IDX, dtype=np.uint8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            current_absolute[square] = (piece.piece_type - 1) + (
                0 if piece.color else 6
            )
    black = board.turn == chess.BLACK
    current = _canonical(current_absolute, black)
    after_rows, changed_rows = [], []
    for depth in range(1, max_depth + 1):
        move = chess.Move.from_uci(snapshots[depth]["bm"])
        after = _canonical(_after_board(current_absolute, board, move), black)
        after_rows.append(after)
        changed_rows.append(after != current)
    return np.stack(after_rows), np.stack(changed_rows)


def prepare_teacher_depth(release_path, output_dir, max_depth=16, board_aux=False):
    """Encode per-depth rank-1 teacher targets for recurrent-depth training.

    Source checksums, target legality, and split overlap by the first four FEN
    fields are verified before anything is published. Rows must complete every
    depth in 1..max_depth; terminal and partially annotated rows are counted and
    excluded. Value targets are unified-vocabulary token ids; the unquantized
    teacher value is kept alongside (+-1 on mate rows, matching the token decode).
    With board_aux, each row additionally carries the canonical square-content
    classes after replaying that depth's teacher move, plus the changed-square
    mask against the current canonical board (horizon-1 future placement).
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
        "kind": "teacher-depth",
        "tokenizer_version": CHESS_TOKENIZER_VERSION,
        "teacher_depth": max_depth,
        "depths": list(range(1, max_depth + 1)),
        "board_aux": board_aux,
        "release_sha256": _checksum(release_path),
        "position_key": "first four FEN fields",
        "legal_actions": "board moves; draw adjudication is external",
        "excluded": {},
    }
    validation_keys = set()
    teacher = None
    for split in ["val", "train"]:
        tokens, moves, value_tokens, values, legal_rows = [], [], [], [], []
        aux_after_rows, aux_changed_rows = [], []
        seen = set()
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
                    targets = _depth_targets(row, max_depth)
                    if targets is None:
                        excluded["incomplete_depths"] += 1
                        continue
                    board = chess.Board(row["fen"])
                    legal = legal_move_indices(board)
                    if targets[0][-1] not in legal:
                        raise ValueError(
                            "Teacher move is illegal in its source position"
                        )
                    tokens.append(encode_board(row["fen"], row.get("repetition", 1)))
                    moves.append(targets[0])
                    value_tokens.append(targets[1])
                    values.append(targets[2])
                    legal_rows.append(legal)
                    if board_aux:
                        aux = _board_aux(row, max_depth)
                        aux_after_rows.append(aux[0])
                        aux_changed_rows.append(aux[1])
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
        for name, rows, padding, dtype in [
            ("tokens", tokens, PAD, np.int16),
            ("legal", legal_rows, -1, np.int16),
        ]:
            array = np.lib.format.open_memmap(
                output_dir / f"{name}_{split}.npy",
                mode="w+",
                dtype=dtype,
                shape=(len(rows), max(map(len, rows))),
            )
            array[:] = padding
            for index, values_row in enumerate(rows):
                array[index, : len(values_row)] = values_row
            array.flush()
        np.save(
            output_dir / f"depth_moves_{split}.npy", np.asarray(moves, dtype=np.int16)
        )
        np.save(
            output_dir / f"depth_value_tokens_{split}.npy",
            np.asarray(value_tokens, dtype=np.int16),
        )
        np.save(
            output_dir / f"depth_values_{split}.npy",
            np.asarray(values, dtype=np.float32),
        )
        if board_aux:
            np.save(output_dir / f"aux_after_{split}.npy", np.stack(aux_after_rows))
            np.save(output_dir / f"aux_changed_{split}.npy", np.stack(aux_changed_rows))
        metadata[f"rows_{split}"] = len(tokens)
        metadata["excluded"][split] = dict(excluded)
    metadata["teacher"] = teacher
    metadata["files"] = {
        path.name: _checksum(path) for path in sorted(output_dir.glob("*.npy"))
    }
    write_manifest(output_dir / "manifest.json", metadata)
    return metadata
