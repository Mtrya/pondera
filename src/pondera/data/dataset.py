"""Read-only training datasets and the shared trace-array contract."""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict
from uuid import uuid4

import numpy as np
from torch.utils.data import Dataset

from pondera.chess.codec import BOARD_TOKENS_MAX
from pondera.chess.vocabulary import CHESS_TOKENIZER_VERSION
from pondera.data.trace import ROLE_COUNT, TRACE_GRAMMAR_VERSION

DATA_FORMAT_VERSION = 6
TRACE_ARRAYS = ("board_tokens", "trace_tokens", "trace_len", "trace_roles")
STATE_ARRAYS = (
    "state_without_board",
    "state_with_board",
    "state_best",
    "state_basis",
    "state_value",
)
DATA_ARRAYS = (*TRACE_ARRAYS, *STATE_ARRAYS, "state_count")


def write_manifest(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
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


def validate_trace_arrays(arrays, rows=None):
    """Check the shared array contract without interpreting individual traces."""
    if not arrays:
        raise ValueError("A Trace dataset needs token arrays")
    row_count = len(next(iter(arrays.values())))
    if rows is not None and row_count != rows:
        raise ValueError("Trace arrays do not match the declared row count")
    for name, array in arrays.items():
        if len(array) != row_count:
            raise ValueError(f"Trace array {name} has a different row count")

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


def load_trace_arrays(directory, names=TRACE_ARRAYS):
    arrays = {
        name: np.load(
            Path(directory) / f"{name}.npy", mmap_mode="r", allow_pickle=False
        )
        for name in names
    }
    validate_trace_arrays(arrays)
    return arrays


class _ShardedTraceArrays:
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
            arrays = load_trace_arrays(shard_dir)
            validate_trace_arrays(arrays, rows=shard["rows"])
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
        arrays = load_trace_arrays(directory)
        validate_trace_arrays(arrays, rows=rows)
        self.cache[shard_index] = arrays
        if len(self.cache) > self.CACHE_SHARDS:
            self.cache.popitem(last=False)
        return arrays

    def get(self, name, index):
        shard, row = self.index[index]
        return self._arrays(shard)[name][row].copy()


def trace_data_identity(directory):
    """The declared shape and generation identity of Trace training input."""
    directory = Path(directory)
    manifest_path = directory / (
        "release.json" if (directory / "release.json").exists() else "manifest.json"
    )
    manifest = load_manifest(manifest_path)
    return {
        "format_version": manifest.get("format_version"),
        "grammar_version": manifest.get("grammar_version"),
        "tokenizer_version": manifest.get("tokenizer_version"),
        "metadata": manifest.get("metadata"),
        "shards": [
            {"path": shard["path"], "rows": shard["rows"]}
            for shard in manifest.get("shards", [])
        ],
        "rows": {split: manifest.get(f"rows_{split}") for split in ("train", "val")},
    }


class TracePositions(Dataset):
    """Read board tokens and their encoded trace rows."""

    def __init__(self, directory, split):
        if split not in {"train", "val"}:
            raise ValueError(f"Unknown trace split: {split}")
        directory = Path(directory)
        manifest_path = directory / (
            "release.json" if (directory / "release.json").exists() else "manifest.json"
        )
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("format_version") != DATA_FORMAT_VERSION:
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
            self.sharded = _ShardedTraceArrays(directory, manifest, split)
            self.manifest = manifest
            return
        self.tokens = np.load(directory / f"board_tokens_{split}.npy", mmap_mode="r")
        self.trace = np.load(directory / f"trace_tokens_{split}.npy", mmap_mode="r")
        self.trace_len = np.load(directory / f"trace_len_{split}.npy", mmap_mode="r")
        self.trace_roles = np.load(
            directory / f"trace_roles_{split}.npy", mmap_mode="r"
        )
        validate_trace_arrays(
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
