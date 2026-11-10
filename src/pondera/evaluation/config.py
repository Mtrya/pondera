"""Checkpoint-evaluation tunables; runtime paths remain command arguments."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("configs/eval.toml")


@dataclass(frozen=True)
class EvalConfig:
    reasoning_tokens: int = 256
    answer_tokens: int = 128
    limit: int = 128
    seed: int = 640
    nodes: int = 10000
    threads: int = 1
    hash_mb: int = 64
    pv_plies: int = 8
    rating_bounds: tuple[int, ...] = (0, 1200, 1600, 2000)


def load_config(path=DEFAULT_CONFIG_PATH):
    with Path(path).open("rb") as handle:
        values = tomllib.load(handle)
    if set(values) - {"generation", "evaluation", "engine", "puzzles"}:
        raise ValueError("Unknown evaluation config section")
    fields = {
        "generation": {"reasoning_tokens", "answer_tokens"},
        "evaluation": {"limit", "seed"},
        "engine": {"nodes", "threads", "hash_mb", "pv_plies"},
        "puzzles": {"rating_bounds"},
    }
    settings = {}
    for section, allowed in fields.items():
        table = values.get(section, {})
        if not isinstance(table, dict) or set(table) - allowed:
            raise ValueError(f"Invalid evaluation config table [{section}]")
        settings.update(table)
    if "rating_bounds" in settings:
        settings["rating_bounds"] = tuple(settings["rating_bounds"])
    config = EvalConfig(**settings)
    for name in (
        "reasoning_tokens",
        "answer_tokens",
        "limit",
        "seed",
        "nodes",
        "threads",
        "hash_mb",
        "pv_plies",
    ):
        value = getattr(config, name)
        minimum = 0 if name in {"reasoning_tokens", "seed", "pv_plies"} else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    if (
        not config.rating_bounds
        or any(type(bound) is not int or bound < 0 for bound in config.rating_bounds)
        or any(a >= b for a, b in zip(config.rating_bounds, config.rating_bounds[1:]))
    ):
        raise ValueError(
            "rating_bounds must be strictly increasing nonnegative integers"
        )
    return config
