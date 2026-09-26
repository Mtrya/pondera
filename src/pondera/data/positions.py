"""Memory-mapped token and move pairs used by training and evaluation."""

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from pondera.chess.board_tokens import BOARD_TOKENIZER_VERSION, MIN_TOKENS
from pondera.chess.tokenize import TOKEN_COUNT, TOKENIZER_VERSION


class TokenizedPositions(Dataset):
    def __init__(
        self, directory, split, tokenizer_version=TOKENIZER_VERSION, legal=False
    ):
        directory = Path(directory)
        with (directory / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        version = manifest.get("tokenizer_version")
        if (
            version not in {TOKENIZER_VERSION, BOARD_TOKENIZER_VERSION}
            or version != tokenizer_version
        ):
            raise ValueError(f"Unsupported dataset tokenizer version: {version}")
        self.tokens = np.load(directory / f"tokens_{split}.npy", mmap_mode="r")
        self.moves = np.load(directory / f"moves_{split}.npy", mmap_mode="r")
        if self.tokens.ndim != 2 or self.tokens.shape[0] != len(self.moves):
            raise ValueError("Token and move arrays have incompatible shapes")
        expected = (
            self.tokens.shape[1] == TOKEN_COUNT and self.tokens.dtype == np.uint8
            if version == TOKENIZER_VERSION
            else self.tokens.shape[1] >= MIN_TOKENS and self.tokens.dtype == np.int16
        )
        if not expected:
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
