"""Explicit model selection for serialized policy configurations."""

from pondera.chess.tokens import CHESS_TOKENIZER_VERSION
from pondera.models.policy import (
    BoardPolicyModel,
    DepthPolicyModel,
    SequencePolicyModel,
)

ARCHITECTURES = {
    "base": BoardPolicyModel,
    "depth": DepthPolicyModel,
    "sequence": SequencePolicyModel,
}

# Keys written by configurations predating the settled representation. Loading
# stays possible only when the recorded values match the settled choices.
SETTLED_OPTIONS = {
    "layout": "pieces",
    "encoding": "joint",
    "orientation": "side_to_move",
    "geometry": "absolute",
    "metadata": "tokens",
    "readout": "single",
}


def model_class(config):
    architecture = config.get("architecture")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown policy architecture: {architecture}")
    # Configurations written before the input format was unified still carry the key.
    if (
        config.get("tokenizer_version", CHESS_TOKENIZER_VERSION)
        != CHESS_TOKENIZER_VERSION
    ):
        raise ValueError("Only the lossless board input format is supported")
    for key, settled in SETTLED_OPTIONS.items():
        if config.get(key, settled) != settled:
            raise ValueError(f"Configuration option {key}={config[key]!r} was retired")
    return ARCHITECTURES[architecture]


def create_model(config):
    arguments = dict(config)
    arguments.pop("architecture", None)
    arguments.pop("tokenizer_version", None)
    for key in SETTLED_OPTIONS:
        arguments.pop(key, None)
    return model_class(config)(**arguments)
