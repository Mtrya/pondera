"""Memory-mapped depth targets and sequence traces used by training."""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from pondera.chess.codec import BOARD_TOKENS_MAX
from pondera.chess.vocabulary import CHESS_TOKENIZER_VERSION
from pondera.data.sequence_validation import (
    SEQUENCE_FORMAT_VERSION,
    load_sequence_arrays,
    validate_sequence_arrays,
)
from pondera.data.trace import TRACE_GRAMMAR_VERSION


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
