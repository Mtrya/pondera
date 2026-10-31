"""Integrity checks and independent replay validation for Sequence releases."""

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import numpy as np

from pondera.chess.codec import BOARD_TOKENS_MAX, encode_board
from pondera.chess.symmetry import reflect_uci
from pondera.chess.vocabulary import (
    CHESS_TOKENIZER_VERSION,
    IDX_TO_UCI_MOVE,
    MOVE,
    PAD,
)
from pondera.data.storage import load_manifest, write_manifest
from pondera.data.trace import ROLE_COUNT, TRACE_GRAMMAR_VERSION, replay

SEQUENCE_FORMAT_VERSION = 2
TRACE_ARRAYS = ("board_tokens", "trace_tokens", "trace_len", "trace_roles")
STATE_ARRAYS = (
    "state_no_tool",
    "state_tool",
    "state_best",
    "state_basis",
    "state_value",
)
SEQUENCE_ARRAYS = (*TRACE_ARRAYS, *STATE_ARRAYS, "state_count")
ROOT_SET_METADATA = frozenset(
    {"root_selection", "root_selections", "roots", "root_sets"}
)


def shard_configuration(metadata):
    return {
        key: value for key, value in metadata.items() if key not in ROOT_SET_METADATA
    }


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_sequence_arrays(arrays, rows=None):
    """Check the shared array contract without interpreting individual traces."""
    if not arrays:
        raise ValueError("A Sequence dataset needs token arrays")
    row_count = len(next(iter(arrays.values())))
    if rows is not None and row_count != rows:
        raise ValueError("Sequence arrays do not match the declared row count")
    for name, array in arrays.items():
        if len(array) != row_count:
            raise ValueError(f"Sequence array {name} has a different row count")

    boards = arrays.get("board_tokens")
    if boards is not None and (
        boards.ndim != 2
        or boards.dtype != np.int16
        or boards.shape[1] > BOARD_TOKENS_MAX
    ):
        raise ValueError("Board token array does not match its declared format")

    trace = arrays.get("trace_tokens")
    if trace is not None and (trace.ndim != 2 or trace.dtype != np.int16):
        raise ValueError("Trace token array does not match its examples")
    lengths = arrays.get("trace_len")
    if lengths is not None:
        if lengths.shape != (row_count,) or lengths.dtype != np.int16:
            raise ValueError("Trace lengths do not match their examples")
        if trace is not None and trace.ndim == 2:
            if np.any(lengths < 1) or np.any(lengths > trace.shape[1]):
                raise ValueError("Trace lengths fall outside their token array")
    roles = arrays.get("trace_roles")
    if roles is not None:
        if trace is not None and roles.shape != trace.shape:
            raise ValueError("Trace roles do not match their examples")
        if roles.dtype != np.int8 or np.any((roles < 0) | (roles >= ROLE_COUNT)):
            raise ValueError("Trace roles do not match their declared format")

    best = arrays.get("state_best")
    if best is not None and (best.ndim != 2 or best.dtype != np.int16):
        raise ValueError("Truncation-state arrays do not match their examples")
    for name in STATE_ARRAYS:
        array = arrays.get(name)
        if array is not None and (
            array.dtype != np.int16 or (best is not None and array.shape != best.shape)
        ):
            raise ValueError(f"Truncation-state array {name} has inconsistent shape")
    counts = arrays.get("state_count")
    if counts is not None:
        if counts.shape != (row_count,) or counts.dtype != np.int16:
            raise ValueError("Truncation-state counts do not match their examples")
        if best is not None and best.ndim == 2:
            if np.any(counts < 0) or np.any(counts > best.shape[1]):
                raise ValueError("Truncation-state counts fall outside their arrays")


def load_sequence_arrays(directory, names=TRACE_ARRAYS):
    arrays = {
        name: np.load(
            Path(directory) / f"{name}.npy", mmap_mode="r", allow_pickle=False
        )
        for name in names
    }
    validate_sequence_arrays(arrays)
    return arrays


def verify_shard_files(release_dir, shard, expected_metadata=None):
    """Verify a published shard and return its manifest, roots, and arrays."""
    directory = Path(release_dir) / shard["path"]
    manifest_path = directory / "manifest.json"
    if sha256(manifest_path) != shard["sha256"]:
        raise ValueError(f"Shard manifest checksum mismatch: {manifest_path}")
    manifest = load_manifest(manifest_path)
    if expected_metadata is not None:
        for key, value in shard_configuration(expected_metadata).items():
            if manifest.get(key) != value:
                raise ValueError(f"Shard metadata {key} differs from its release")
    if manifest["rows"] != shard["rows"]:
        raise ValueError(f"Shard row count differs from its release: {directory}")
    expected_files = {f"{name}.npy" for name in SEQUENCE_ARRAYS} | {"roots.jsonl"}
    if set(manifest.get("files", {})) != expected_files:
        raise ValueError(f"Shard file inventory is incomplete: {directory}")
    for name, expected in manifest["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Shard file checksum mismatch: {directory / name}")
    roots = [
        json.loads(line)
        for line in (directory / "roots.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(roots) != manifest["rows"]:
        raise ValueError(f"Shard roots do not match its manifest: {directory}")
    arrays = load_sequence_arrays(directory, SEQUENCE_ARRAYS)
    validate_sequence_arrays(arrays, rows=len(roots))
    return manifest, roots, arrays


def _validate_shard_rows(release_dir, shard, teachers, expected_metadata=None):
    _, rows, arrays = verify_shard_files(release_dir, shard, expected_metadata)
    identities = []
    counts = Counter()
    state_total = 0
    board_responses = 0
    max_trace = 0
    for index, row in enumerate(rows):
        if row["teacher"] not in teachers:
            raise ValueError(f"Unknown teacher in shard row: {row['teacher']}")
        if row["split"] not in {"train", "val"}:
            raise ValueError(f"Unknown split in shard row: {row['split']}")
        board = chess.Board(row["fen"])
        if board.turn != (row["turn"] == "white"):
            raise ValueError(f"Root turn disagrees with FEN: {row['root_id']}")
        encoded = encode_board(row["fen"], row["repetition"])
        board_tokens = arrays["board_tokens"][index]
        if not np.array_equal(board_tokens[: len(encoded)], encoded) or not np.all(
            board_tokens[len(encoded) :] == PAD
        ):
            raise ValueError(f"Board tokens disagree with FEN: {row['root_id']}")

        length = int(arrays["trace_len"][index])
        tokens = arrays["trace_tokens"][index, :length]
        if not np.all(arrays["trace_tokens"][index, length:] == PAD) or not np.all(
            arrays["trace_roles"][index, length:] == 0
        ):
            raise ValueError(f"Trace padding is invalid: {row['root_id']}")
        program = replay(tokens, row["fen"], row["repetition"])
        if program.final_move_index != length - 1 or not np.array_equal(
            program.roles, arrays["trace_roles"][index, :length]
        ):
            raise ValueError(f"Trace roles disagree with replay: {row['root_id']}")
        max_trace = max(max_trace, length)

        board_spans = [
            (step.start, step.end)
            for step in program.steps
            if step.operation == "board"
        ]
        board_responses += len(board_spans)
        no_tool_tokens = list(tokens)
        for start, end in reversed(board_spans):
            del no_tool_tokens[start:end]
        no_tool = replay(no_tool_tokens, row["fen"], row["repetition"])
        expected_boundaries = []
        operation = 0
        previous_no_tool_end = 0
        for step in program.steps:
            if step.operation == "board":
                expected_boundaries.append((step.end, previous_no_tool_end))
                continue
            no_tool_step = no_tool.steps[operation]
            if (
                step.position != no_tool_step.position
                or step.uci != no_tool_step.uci
                or step.evaluation != no_tool_step.evaluation
                or step.mate != no_tool_step.mate
            ):
                raise ValueError(f"Tool and no-tool traces differ: {row['root_id']}")
            previous_no_tool_end = no_tool_step.end
            if step.operation != "final":
                expected_boundaries.append((step.end, no_tool_step.end))
            operation += 1
        if operation != len(no_tool.steps):
            raise ValueError(f"Tool and no-tool operations differ: {row['root_id']}")
        expected_boundaries.sort()

        count = int(arrays["state_count"][index])
        if count != len(expected_boundaries):
            raise ValueError(f"Truncation-state count is invalid: {row['root_id']}")
        actual_boundaries = list(
            zip(
                map(int, arrays["state_tool"][index, :count]),
                map(int, arrays["state_no_tool"][index, :count]),
            )
        )
        if actual_boundaries != expected_boundaries:
            raise ValueError(f"Truncation boundaries are invalid: {row['root_id']}")
        for name in STATE_ARRAYS:
            if not np.all(arrays[name][index, count:] == -1):
                raise ValueError(
                    f"Truncation-state padding is invalid: {row['root_id']}"
                )
        basis = arrays["state_basis"][index, :count]
        if np.any((basis < 0) | (basis > 2)):
            raise ValueError(f"Truncation-state basis is invalid: {row['root_id']}")
        mirrored = board.turn == chess.BLACK
        for token in arrays["state_best"][index, :count]:
            token = int(token)
            if not MOVE <= token < MOVE + len(IDX_TO_UCI_MOVE):
                raise ValueError(f"Truncation-state move is invalid: {row['root_id']}")
            uci = IDX_TO_UCI_MOVE[token - MOVE]
            if mirrored:
                uci = reflect_uci(uci)
            if chess.Move.from_uci(uci) not in board.legal_moves:
                raise ValueError(f"Truncation-state move is illegal: {row['root_id']}")
        identities.append(
            (row["root_id"], row["teacher"], row["fen"], row["repetition"])
        )
        counts[row["split"]] += 1
        counts[row["teacher"]] += 1
        state_total += count
    return {
        "identities": identities,
        "counts": counts,
        "states": state_total,
        "board_responses": board_responses,
        "max_trace": max_trace,
    }


def validate_release(directory, roots=None, workers=8):
    """Validate every file and replay every row of a complete sharded release."""
    if workers < 1:
        raise ValueError("Validation workers must be positive")
    directory = Path(directory)
    release = load_manifest(directory / "release.json")
    expected_metadata = {
        "format_version": SEQUENCE_FORMAT_VERSION,
        "grammar_version": TRACE_GRAMMAR_VERSION,
        "tokenizer_version": CHESS_TOKENIZER_VERSION,
        "kind": "teacher-sequence",
        "role_count": ROLE_COUNT,
    }
    for key, value in expected_metadata.items():
        if release.get(key) != value:
            raise ValueError(f"Release metadata {key} is unsupported")
    expected_shard_metadata = {**expected_metadata, **release["metadata"]}
    teachers = release["metadata"]["teachers"]
    if not teachers or len(set(teachers)) != len(teachers):
        raise ValueError("Release teacher names must be nonempty and unique")
    complete = release["complete_root_ids"]
    if len(set(complete)) != len(complete):
        raise ValueError("Release root ids must be unique")
    expected = {row["root_id"]: row for row in roots} if roots is not None else None
    seen = set()
    seen_roots = set()
    counts = Counter()
    states = 0
    board_responses = 0
    max_trace = 0

    def validate(shard):
        return _validate_shard_rows(directory, shard, teachers, expected_shard_metadata)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(validate, release["shards"]):
            for root_id, teacher, fen, repetition in result["identities"]:
                key = (root_id, teacher)
                if key in seen:
                    raise ValueError(f"Duplicate teacher row: {key}")
                seen.add(key)
                seen_roots.add(root_id)
                if expected is not None:
                    if root_id not in expected:
                        raise ValueError(
                            f"Release contains an unselected root: {root_id}"
                        )
                    root = expected[root_id]
                    if fen != root["fen"] or repetition != root["repetition"]:
                        raise ValueError(f"Release root provenance changed: {root_id}")
            counts.update(result["counts"])
            states += result["states"]
            board_responses += result["board_responses"]
            max_trace = max(max_trace, result["max_trace"])
    if seen_roots != set(complete) or len(seen) != len(complete) * len(teachers):
        raise ValueError("Release rows do not match its roots and teachers")
    if expected is not None:
        wanted = {(root_id, teacher) for root_id in expected for teacher in teachers}
        if seen != wanted or set(complete) != set(expected):
            raise ValueError("Release does not contain exactly the selected roots")
    result = {
        "ok": True,
        "roots": len(complete),
        "rows": len(seen),
        "states": states,
        "board_responses": board_responses,
        "max_trace": max_trace,
        "counts": dict(counts),
        "shards": len(release["shards"]),
    }
    write_manifest(directory / "validation.json", result)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--roots", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    roots = None
    if args.roots is not None:
        with args.roots.open(encoding="utf-8") as handle:
            roots = [json.loads(line) for line in handle if line.strip()][
                args.start : args.end
            ]
    result = validate_release(args.release, roots, args.workers)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
