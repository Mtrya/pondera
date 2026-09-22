"""Memory-mapped token and move pairs used by training and evaluation."""

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from pondera.chess.tokenize import TOKEN_COUNT, TOKENIZER_VERSION


class TokenizedPositions(Dataset):
    def __init__(self, directory, split):
        directory = Path(directory)
        with (directory / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        version = manifest.get("tokenizer_version")
        if version != TOKENIZER_VERSION:
            raise ValueError(f"Unsupported dataset tokenizer version: {version}")
        self.tokens = np.load(directory / f"tokens_{split}.npy", mmap_mode="r")
        self.moves = np.load(directory / f"moves_{split}.npy", mmap_mode="r")
        if self.tokens.shape != (len(self.moves), TOKEN_COUNT):
            raise ValueError("Token and move arrays have incompatible shapes")

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, index):
        return self.tokens[index].copy(), int(self.moves[index])
