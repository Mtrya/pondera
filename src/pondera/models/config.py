"""Transformer block dimensions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockConfig:
    hidden_size: int = 256
    intermediate_size: int = 768
    num_heads: int = 8
    dropout: float = 0.0

    def __post_init__(self):
        if min(self.hidden_size, self.intermediate_size, self.num_heads) < 1:
            raise ValueError("Block dimensions must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("Invalid dropout")
