"""Configuration for token backbones and independently repeated layer groups."""

import math
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
    mode: str = "fixed"
    max_repetitions: int = 1
    min_repetitions: int = 1
    scope: str = "individual"
    tolerance: float = 1e-3
    stop_threshold: float = 0.5

    def __post_init__(self):
        if self.mode not in {"fixed", "convergence", "learned"}:
            raise ValueError("Unknown repetition controller")
        if not 1 <= self.min_repetitions <= self.max_repetitions:
            raise ValueError("Invalid repetition bounds")
        if self.scope not in {"individual", "shared"}:
            raise ValueError("Repetition scope must be individual or shared")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("Convergence tolerance must be finite and nonnegative")
        if not 0 < self.stop_threshold < 1:
            raise ValueError("Stop threshold must lie strictly between zero and one")


@dataclass(frozen=True)
class CoreConfig:
    layers: int = 1
    repetition: RepetitionConfig = field(default_factory=RepetitionConfig)
    reinject_input: bool = False
    condition_on_iteration: bool = False

    def __post_init__(self):
        if self.layers < 1:
            raise ValueError("A core needs at least one layer")
        if isinstance(self.repetition, dict):
            object.__setattr__(self, "repetition", RepetitionConfig(**self.repetition))
