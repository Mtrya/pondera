"""Configuration for token backbones and independently repeated layer groups."""

from dataclasses import dataclass, field


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


@dataclass(frozen=True)
class RepetitionConfig:
    max_repetitions: int = 1

    def __post_init__(self):
        if self.max_repetitions < 1:
            raise ValueError("max_repetitions must be positive")


@dataclass(frozen=True)
class CoreConfig:
    layers: int = 1
    repetition: RepetitionConfig = field(default_factory=RepetitionConfig)

    def __post_init__(self):
        if self.layers < 1:
            raise ValueError("A core needs at least one layer")
        if isinstance(self.repetition, dict):
            object.__setattr__(self, "repetition", RepetitionConfig(**self.repetition))
