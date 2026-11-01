"""Read-only training datasets and the shared sequence-array contract."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import numpy as np
from torch.utils.data import Dataset

from pondera.chess.codec import BOARD_TOKENS_MAX
from pondera.chess.vocabulary import CHESS_TOKENIZER_VERSION
from pondera.data.trace import ROLE_COUNT, TRACE_GRAMMAR_VERSION

SEQUENCE_FORMAT_VERSION = 5
TRACE_ARRAYS = ("board_tokens", "trace_tokens", "trace_len", "trace_roles")
STATE_ARRAYS = (
    "state_no_tool",
    "state_tool",
    "state_best",
    "state_basis",
    "state_value",
)
SEQUENCE_ARRAYS = (*TRACE_ARRAYS, *STATE_ARRAYS, "state_count")


def sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def content_sha256(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_manifest(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_manifest(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
        if lengths.shape != (row_count,) or lengths.dtype != np.int32:
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
    if best is not None and (best.ndim != 2 or best.dtype != np.int32):
        raise ValueError("Truncation-state arrays do not match their examples")
    for name in STATE_ARRAYS:
        array = arrays.get(name)
        if array is not None and (
            array.dtype != np.int32 or (best is not None and array.shape != best.shape)
        ):
            raise ValueError(f"Truncation-state array {name} has inconsistent shape")
    counts = arrays.get("state_count")
    if counts is not None:
        if counts.shape != (row_count,) or counts.dtype != np.int32:
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


class DepthPositions(Dataset):
    """Board tokens with per-depth rank-1 teacher targets (kind=teacher-depth)."""

    def __init__(self, directory, split, legal=False, board_aux=False):
        directory = Path(directory)
        with (directory / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("kind") != "teacher-depth":
            raise ValueError("Not a per-depth teacher dataset")
        if manifest.get("tokenizer_version") != CHESS_TOKENIZER_VERSION:
            raise ValueError(
                f"Unsupported dataset tokenizer version: {manifest.get('tokenizer_version')}"
            )
        self.depths = manifest["depths"]
        self.tokens = np.load(directory / f"tokens_{split}.npy", mmap_mode="r")
        self.moves = np.load(directory / f"depth_moves_{split}.npy", mmap_mode="r")
        self.value_tokens = np.load(
            directory / f"depth_value_tokens_{split}.npy", mmap_mode="r"
        )
        self.values = np.load(directory / f"depth_values_{split}.npy", mmap_mode="r")
        rows, depth_count = len(self.tokens), len(self.depths)
        if (
            self.tokens.ndim != 2
            or self.tokens.dtype != np.int16
            or self.tokens.shape[1] > BOARD_TOKENS_MAX
        ):
            raise ValueError("Token array does not match its declared format")
        for name, array, dtype in (
            ("moves", self.moves, np.int16),
            ("value_tokens", self.value_tokens, np.int16),
            ("values", self.values, np.float32),
        ):
            if array.shape != (rows, depth_count) or array.dtype != dtype:
                raise ValueError(f"Depth {name} array does not match its examples")
        self.manifest = manifest
        self.legal = None
        if legal:
            self.legal = np.load(directory / f"legal_{split}.npy", mmap_mode="r")
            if self.legal.ndim != 2 or len(self.legal) != rows:
                raise ValueError("Legal action array does not match the examples")
        self.aux_after = self.aux_changed = None
        if board_aux:
            if not manifest.get("board_aux"):
                raise ValueError("Dataset was not encoded with board aux targets")
            self.aux_after = np.load(
                directory / f"aux_after_{split}.npy", mmap_mode="r"
            )
            self.aux_changed = np.load(
                directory / f"aux_changed_{split}.npy", mmap_mode="r"
            )
            if (
                self.aux_after.shape != (rows, depth_count, 64)
                or self.aux_after.dtype != np.uint8
                or self.aux_changed.shape != (rows, depth_count, 64)
                or self.aux_changed.dtype != np.bool_
            ):
                raise ValueError("Board aux arrays do not match the examples")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        row = (
            self.tokens[index].copy(),
            self.moves[index].copy(),
            self.value_tokens[index].copy(),
            self.values[index].copy(),
        )
        if self.legal is not None:
            row = (*row, self.legal[index].copy())
        if self.aux_after is not None:
            row = (*row, self.aux_after[index].copy(), self.aux_changed[index].copy())
        return row


class _ShardedSequenceArrays:
    """Lazily memory-map shards so open descriptors stay bounded.

    Every shard is validated once at construction. Arrays are then reopened on
    demand through a small LRU cache; holding all four mappings of every shard
    open would exhaust the descriptor limit of a large release.
    """

    CACHE_SHARDS = 32

    def __init__(self, directory, manifest, split):
        self.index = []
        self.shards = []
        self.cache = OrderedDict()
        for shard in manifest["shards"]:
            shard_dir = directory / shard["path"]
            arrays = load_sequence_arrays(shard_dir)
            validate_sequence_arrays(arrays, rows=shard["rows"])
            del arrays
            with (shard_dir / "roots.jsonl").open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            if len(rows) != shard["rows"]:
                raise ValueError(f"Shard roots do not match its release: {shard_dir}")
            shard_index = len(self.shards)
            self.shards.append((shard_dir, shard["rows"]))
            for row_index, row in enumerate(rows):
                if row["split"] not in {"train", "val"}:
                    raise ValueError(f"Unknown shard split: {row['split']}")
                if row["split"] == split:
                    self.index.append((shard_index, row_index))

    def _arrays(self, shard_index):
        arrays = self.cache.get(shard_index)
        if arrays is not None:
            self.cache.move_to_end(shard_index)
            return arrays
        directory, rows = self.shards[shard_index]
        arrays = load_sequence_arrays(directory)
        validate_sequence_arrays(arrays, rows=rows)
        self.cache[shard_index] = arrays
        if len(self.cache) > self.CACHE_SHARDS:
            self.cache.popitem(last=False)
        return arrays

    def get(self, name, index):
        shard, row = self.index[index]
        return self._arrays(shard)[name][row].copy()


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
    return content_sha256({"manifest": manifest, "files": files})


class SequencePositions(Dataset):
    """Board tokens with an encoded sequence trace (kind=teacher-sequence).

    Rows are (boards, trace tokens, trace length, trace roles). Every trace
    ends with ``<end>``; the FINAL decision carries ROLE_FINAL_MOVE and the
    label suffix continues after it. Positions past the stored length are
    padding.
    """

    def __init__(self, directory, split):
        if split not in {"train", "val"}:
            raise ValueError(f"Unknown sequence split: {split}")
        directory = Path(directory)
        manifest_path = directory / (
            "release.json" if (directory / "release.json").exists() else "manifest.json"
        )
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("kind") != "teacher-sequence":
            raise ValueError("Not a sequence trace dataset")
        if manifest.get("format_version") != SEQUENCE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported dataset format version: {manifest.get('format_version')}"
            )
        if manifest.get("tokenizer_version") != CHESS_TOKENIZER_VERSION:
            raise ValueError(
                f"Unsupported dataset tokenizer version: {manifest.get('tokenizer_version')}"
            )
        if manifest.get("grammar_version") != TRACE_GRAMMAR_VERSION:
            raise ValueError(
                f"Unsupported trace grammar version: {manifest.get('grammar_version')}"
            )
        if "shards" in manifest:
            self.sharded = _ShardedSequenceArrays(directory, manifest, split)
            self.manifest = manifest
            return
        self.tokens = np.load(directory / f"board_tokens_{split}.npy", mmap_mode="r")
        self.trace = np.load(directory / f"trace_tokens_{split}.npy", mmap_mode="r")
        self.trace_len = np.load(directory / f"trace_len_{split}.npy", mmap_mode="r")
        self.trace_roles = np.load(
            directory / f"trace_roles_{split}.npy", mmap_mode="r"
        )
        validate_sequence_arrays(
            {
                "board_tokens": self.tokens,
                "trace_tokens": self.trace,
                "trace_len": self.trace_len,
                "trace_roles": self.trace_roles,
            },
            rows=manifest.get(f"rows_{split}"),
        )
        self.manifest = manifest

    def __len__(self):
        return len(self.sharded.index) if hasattr(self, "sharded") else len(self.tokens)

    def __getitem__(self, index):
        if hasattr(self, "sharded"):
            return (
                self.sharded.get("board_tokens", index),
                self.sharded.get("trace_tokens", index),
                int(self.sharded.get("trace_len", index)),
                self.sharded.get("trace_roles", index),
            )
        return (
            self.tokens[index].copy(),
            self.trace[index].copy(),
            int(self.trace_len[index]),
            self.trace_roles[index].copy(),
        )
