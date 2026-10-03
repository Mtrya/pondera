"""Memory-mapped board tokens, moves, and legal action sets used by training."""

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from pondera.chess.tokens import CHESS_TOKENIZER_VERSION, MIN_TOKENS


class TokenizedPositions(Dataset):
    def __init__(self, directory, split, legal=False):
        directory = Path(directory)
        with (directory / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        version = manifest.get("tokenizer_version")
        if version != CHESS_TOKENIZER_VERSION:
            raise ValueError(f"Unsupported dataset tokenizer version: {version}")
        self.tokens = np.load(directory / f"tokens_{split}.npy", mmap_mode="r")
        self.moves = np.load(directory / f"moves_{split}.npy", mmap_mode="r")
        if self.tokens.ndim != 2 or self.tokens.shape[0] != len(self.moves):
            raise ValueError("Token and move arrays have incompatible shapes")
        if self.tokens.shape[1] < MIN_TOKENS or self.tokens.dtype != np.int16:
            raise ValueError("Token array does not match its declared format")
        self.manifest = manifest
        self.legal = None
        if legal:
            self.legal = np.load(directory / f"legal_{split}.npy", mmap_mode="r")
            if (
                self.legal.ndim != 2
                or len(self.legal) != len(self.moves)
                or self.legal.dtype != np.int16
            ):
                raise ValueError("Legal action array does not match the examples")

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, index):
        row = self.tokens[index].copy(), int(self.moves[index])
        return row if self.legal is None else (*row, self.legal[index].copy())


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
            or self.tokens.shape[1] < MIN_TOKENS
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
            self.aux_after = np.load(directory / f"aux_after_{split}.npy", mmap_mode="r")
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


class SequencePositions(Dataset):
    """Board tokens with an encoded sequence trace (kind=teacher-sequence).

    Rows are (boards, trace tokens, trace length, segment bounds). Bounds are
    ordered [start, end) pairs whose last entry is the FINAL segment; padding is
    (-1, -1) and only the leading LINE segments may be reordered.
    """

    def __init__(self, directory, split):
        directory = Path(directory)
        with (directory / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("kind") != "teacher-sequence":
            raise ValueError("Not a sequence trace dataset")
        if manifest.get("tokenizer_version") != CHESS_TOKENIZER_VERSION:
            raise ValueError(
                f"Unsupported dataset tokenizer version: {manifest.get('tokenizer_version')}"
            )
        self.level = manifest["level"]
        self.tokens = np.load(directory / f"board_tokens_{split}.npy", mmap_mode="r")
        self.trace = np.load(directory / f"trace_tokens_{split}.npy", mmap_mode="r")
        self.trace_len = np.load(directory / f"trace_len_{split}.npy", mmap_mode="r")
        self.line_bounds = np.load(
            directory / f"line_bounds_{split}.npy", mmap_mode="r"
        )
        rows = len(self.tokens)
        if (
            self.tokens.ndim != 2
            or self.tokens.dtype != np.int16
            or self.tokens.shape[1] < MIN_TOKENS
        ):
            raise ValueError("Board token array does not match its declared format")
        if (
            self.trace.ndim != 2
            or self.trace.dtype != np.int16
            or self.trace.shape[0] != rows
        ):
            raise ValueError("Trace token array does not match its examples")
        if self.trace_len.shape != (rows,) or self.trace_len.dtype != np.int16:
            raise ValueError("Trace lengths do not match their examples")
        if (
            self.line_bounds.ndim != 3
            or self.line_bounds.shape[0] != rows
            or self.line_bounds.shape[2] != 2
            or self.line_bounds.shape[1] != manifest["max_lines"]
            or self.line_bounds.dtype != np.int16
        ):
            raise ValueError("Segment bounds do not match their examples")
        self.manifest = manifest

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return (
            self.tokens[index].copy(),
            self.trace[index].copy(),
            int(self.trace_len[index]),
            self.line_bounds[index].copy(),
        )
