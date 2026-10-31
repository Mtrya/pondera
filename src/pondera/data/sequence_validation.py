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
    DEPTH,
    IDX_TO_UCI_MOVE,
    MOVE,
    PAD,
    VALUE,
)
from pondera.data.storage import load_manifest, write_manifest
from pondera.data.trace import ROLE_COUNT, TRACE_GRAMMAR_VERSION, replay

SEQUENCE_FORMAT_VERSION = 3
TRACE_ARRAYS = ("board_tokens", "trace_tokens", "trace_len", "trace_roles")
STATE_ARRAYS = (
    "state_no_tool",
    "state_tool",
    "state_best",
    "state_basis",
    "state_value",
)
SEQUENCE_ARRAYS = (*TRACE_ARRAYS, *STATE_ARRAYS, "state_count")
# The label region after FINAL is not a thinking boundary; truncation states
# stop at the last thought operation.
FINAL_SUFFIX_OPS = frozenset({"final", "final_evaluate", "final_pv", "end"})
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


def _content_sha256(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sequence_data_identity(directory):
    """Verify and fingerprint the files that define Sequence training input."""
    directory = Path(directory)
    manifest_path = directory / (
        "release.json" if (directory / "release.json").exists() else "manifest.json"
    )
    manifest = load_manifest(manifest_path)
    files = {}
    if "shards" in manifest:
        for shard in manifest["shards"]:
            shard_dir = directory / shard["path"]
            manifest_path = shard_dir / "manifest.json"
            manifest_sha = sha256(manifest_path)
            if manifest_sha != shard["sha256"]:
                raise ValueError(f"Shard manifest checksum mismatch: {manifest_path}")
            shard_manifest = load_manifest(manifest_path)
            files[manifest_path.relative_to(directory).as_posix()] = manifest_sha
            filenames = [f"{name}.npy" for name in TRACE_ARRAYS] + ["roots.jsonl"]
            for filename in filenames:
                expected = shard_manifest.get("files", {}).get(filename)
                if expected is None:
                    raise ValueError(
                        f"Shard has no checksum for {filename}: {shard_dir}"
                    )
                path = shard_dir / filename
                actual = sha256(path)
                if actual != expected:
                    raise ValueError(f"Training input checksum mismatch: {path}")
                files[path.relative_to(directory).as_posix()] = actual
    else:
        if not manifest.get("files"):
            raise ValueError("Sequence dataset manifest has no input checksums")
        for filename, expected in manifest["files"].items():
            path = directory / filename
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"Training input checksum mismatch: {path}")
            files[filename] = actual
    return _content_sha256({"manifest": manifest, "files": files})


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


def _validate_shard_rows(
    release_dir, shard, teachers, expected_metadata=None, expected_roots=None
):
    manifest, rows, arrays = verify_shard_files(release_dir, shard, expected_metadata)
    teacher_width = int(manifest.get("teacher_width", 1))
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
        if expected_roots is not None:
            expected = expected_roots.get(row["root_id"])
            if expected is None:
                raise ValueError(
                    f"Release contains an unselected root: {row['root_id']}"
                )
            for name in ROOT_RECORD_FIELDS:
                if row.get(name) != expected.get(name):
                    raise ValueError(
                        f"Release root {name} differs from its selection: {row['root_id']}"
                    )
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
        program = replay(
            tokens,
            row["fen"],
            row["repetition"],
            teacher_width=teacher_width,
        )
        if not np.array_equal(program.roles, arrays["trace_roles"][index, :length]):
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
        no_tool = replay(
            no_tool_tokens,
            row["fen"],
            row["repetition"],
            teacher_width=teacher_width,
        )
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
            if step.operation not in FINAL_SUFFIX_OPS:
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
        rounds = int(np.count_nonzero(np.asarray(tokens) == DEPTH))
        if np.any((basis < 0) | (basis > rounds)):
            raise ValueError(f"Truncation-state basis is invalid: {row['root_id']}")
        mirrored = board.turn == chess.BLACK
        best = arrays["state_best"][index, :count]
        values = arrays["state_value"][index, :count]
        for move_token, value_token in zip(best, values):
            move_token, value_token = int(move_token), int(value_token)
            if move_token == -1 or value_token == -1:
                if move_token != -1 or value_token != -1:
                    raise ValueError(
                        f"Truncation-state champion is half unset: {row['root_id']}"
                    )
                continue
            if not MOVE <= move_token < MOVE + len(IDX_TO_UCI_MOVE):
                raise ValueError(f"Truncation-state move is invalid: {row['root_id']}")
            if not VALUE <= value_token < PAD:
                raise ValueError(f"Truncation-state value is invalid: {row['root_id']}")
            uci = IDX_TO_UCI_MOVE[move_token - MOVE]
            if mirrored:
                uci = reflect_uci(uci)
            if chess.Move.from_uci(uci) not in board.legal_moves:
                raise ValueError(f"Truncation-state move is illegal: {row['root_id']}")
        identities.append((row["root_id"], row["teacher"]))
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


SEARCH_FIELDS = ("depth", "multipv", "threads", "hash_mb", "max_time")
TEACHER_SPEC_FIELDS = frozenset(
    {
        "kind",
        "max_depth",
        "max_nodes",
        "width",
        "value_floor",
        "near_value",
        "max_candidates",
    }
)
ROOT_RECORD_FIELDS = ("split", "fen", "repetition", "turn", "d16", "source")


def _valid_root_set(identity):
    if not isinstance(identity, dict) or set(identity) != {"rows", "sha256"}:
        return False
    rows = identity["rows"]
    digest = identity["sha256"]
    return (
        isinstance(rows, int)
        and not isinstance(rows, bool)
        and rows > 0
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def _release_identity_contract(release, root_count):
    metadata = release.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Release has no synthesis identity")
    teachers = metadata.get("teachers")
    if (
        not isinstance(teachers, list)
        or not teachers
        or not all(isinstance(name, str) for name in teachers)
        or len(set(teachers)) != len(teachers)
    ):
        raise ValueError("Release teacher names must be nonempty and unique")
    search = metadata.get("search")
    if not isinstance(search, dict) or any(
        name not in search for name in SEARCH_FIELDS
    ):
        raise ValueError("Release identity has incomplete search settings")
    specs = metadata.get("teacher_specs")
    if not isinstance(specs, dict) or set(specs) != set(teachers):
        raise ValueError("Release identity does not cover every teacher")
    for name in teachers:
        spec = specs[name]
        if not isinstance(spec, dict) or set(spec) != TEACHER_SPEC_FIELDS:
            raise ValueError(f"Teacher identity is incomplete: {name}")
        if spec["kind"] not in {"fixed", "adaptive"}:
            raise ValueError(f"Teacher identity has an unknown kind: {name}")
    if ("roots" in metadata) == ("root_sets" in metadata):
        raise ValueError("Release identity needs one roots or root_sets record")
    if "roots" in metadata:
        root_sets = [metadata["roots"]]
    else:
        root_sets = metadata["root_sets"]
        if not isinstance(root_sets, list) or not root_sets:
            raise ValueError("Release root_sets must be a nonempty list")
    if not all(_valid_root_set(root_set) for root_set in root_sets):
        raise ValueError("Release root identity is malformed")
    if sum(root_set["rows"] for root_set in root_sets) != root_count:
        raise ValueError("Release root identity does not match its root count")
    for key, value in metadata.items():
        if release.get(key) != value:
            raise ValueError(f"Release metadata {key} differs from its identity")
    return metadata


def validate_release_identity(release):
    complete = release.get("complete_root_ids")
    if (
        not isinstance(complete, list)
        or not complete
        or len(set(complete)) != len(complete)
    ):
        raise ValueError("Release root ids must be nonempty and unique")
    return _release_identity_contract(release, len(complete))


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
    complete = release["complete_root_ids"]
    metadata = validate_release_identity(release)
    expected_shard_metadata = {**expected_metadata, **metadata}
    teachers = metadata["teachers"]
    expected = None
    if roots is not None:
        root_records = list(roots)
        expected = {row["root_id"]: row for row in root_records}
        if len(expected) != len(root_records):
            raise ValueError("External root records contain duplicate root ids")
        if "roots" in metadata:
            identity = metadata["roots"]
            if (
                len(root_records) != identity["rows"]
                or _content_sha256(root_records) != identity["sha256"]
            ):
                raise ValueError("External root records do not match the release hash")
    seen = set()
    seen_roots = set()
    counts = Counter()
    states = 0
    board_responses = 0
    max_trace = 0

    def validate(shard):
        return _validate_shard_rows(
            directory, shard, teachers, expected_shard_metadata, expected
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(validate, release["shards"]):
            for root_id, teacher in result["identities"]:
                key = (root_id, teacher)
                if key in seen:
                    raise ValueError(f"Duplicate teacher row: {key}")
                seen.add(key)
                seen_roots.add(root_id)
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
