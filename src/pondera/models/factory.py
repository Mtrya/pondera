"""Explicit model selection for serialized policy configurations."""

from pondera.chess.board_tokens import BOARD_TOKENIZER_VERSION
from pondera.models.policy import BoardPolicyModel, DepthPolicyModel

ARCHITECTURES = {"base": BoardPolicyModel, "depth": DepthPolicyModel}


def model_class(config):
    architecture = config.get("architecture")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown policy architecture: {architecture}")
    # Configurations written before the input format was unified still carry the key.
    if (
        config.get("tokenizer_version", BOARD_TOKENIZER_VERSION)
        != BOARD_TOKENIZER_VERSION
    ):
        raise ValueError("Only the lossless board input format is supported")
    return ARCHITECTURES[architecture]


def create_model(config):
    arguments = dict(config)
    arguments.pop("architecture", None)
    arguments.pop("tokenizer_version", None)
    return model_class(config)(**arguments)
