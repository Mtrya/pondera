"""Training recipe settings; dataset and runtime locations stay in arguments."""

import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("configs/train.toml")
_MODEL_DIMENSIONS = (
    "num_blocks",
    "hidden_size",
    "intermediate_size",
    "num_heads",
    "num_kv_heads",
)


@dataclass
class TrainConfig:
    muon_lr: float = 0.02
    adamw_lr: float = 3e-4
    weight_decay: float = 0.0
    microbatch_tokens: int = 131_072
    update_tokens: int = 524_288
    presentations: float = 1.0
    documents: int | None = None
    max_document_tokens: int | None = None
    updates: int | None = None
    val_every: int = 10
    checkpoint_every: int = 50
    checkpoint_keep_last: int = 3
    seed: int = 640


def _table(values, name, allowed):
    if not isinstance(values, dict):
        raise ValueError(f"[{name}] must be a table")
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown config key in [{name}]: {sorted(unknown)[0]}")
    return values


def _integer(section, name, value, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"[{section}] {name} must be an integer at least {minimum}")


def load_config(path=DEFAULT_CONFIG_PATH):
    """Read model parameters and training settings from one TOML recipe.

    Optional training fields inherit TrainConfig defaults. Vocabulary size
    is derived from encoded data, so it cannot be specified in the recipe.
    """
    with Path(path).open("rb") as handle:
        values = tomllib.load(handle)
    _table(values, "root", {"model", "training"})
    for section in ("model", "training"):
        if section not in values:
            raise ValueError(f"Missing config section: [{section}]")
    model = _table(values["model"], "model", {"architecture", *_MODEL_DIMENSIONS})
    missing = {"architecture", *_MODEL_DIMENSIONS} - set(model)
    if missing:
        raise ValueError(f"Missing config key in [model]: {sorted(missing)[0]}")
    if model["architecture"] not in ("vanilla", "linear"):
        raise ValueError("[model] architecture must be vanilla or linear")
    for name in _MODEL_DIMENSIONS:
        _integer("model", name, model[name], 1)
    if (
        model["hidden_size"] % model["num_heads"]
        or model["num_heads"] % model["num_kv_heads"]
    ):
        raise ValueError("[model] head counts must divide each other and hidden_size")

    training = _table(
        values["training"], "training", {field.name for field in fields(TrainConfig)}
    )
    config = TrainConfig(**training)
    for name in (
        "microbatch_tokens",
        "update_tokens",
        "documents",
        "max_document_tokens",
        "updates",
        "val_every",
        "checkpoint_every",
        "checkpoint_keep_last",
        "seed",
    ):
        value = getattr(config, name)
        if value is not None:
            _integer(
                "training",
                name,
                value,
                0 if name in {"val_every", "checkpoint_every", "seed"} else 1,
            )
    for name in ("muon_lr", "adamw_lr", "weight_decay", "presentations"):
        value = getattr(config, name)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or (name == "presentations" and value == 0)
        ):
            raise ValueError(
                f"[training] {name} must be finite and {'positive' if name == 'presentations' else 'nonnegative'}"
            )
    return model, config
