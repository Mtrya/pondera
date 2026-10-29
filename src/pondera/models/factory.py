"""Explicit model selection for serialized policy configurations."""

from pondera.chess.vocabulary import CHESS_TOKENIZER_VERSION
from pondera.models.policy import DepthPolicyModel, SequencePolicyModel

ARCHITECTURES = {
    "depth": DepthPolicyModel,
    "sequence": SequencePolicyModel,
}


def model_class(config):
    architecture = config.get("architecture")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown policy architecture: {architecture}")
    if (
        config.get("tokenizer_version", CHESS_TOKENIZER_VERSION)
        != CHESS_TOKENIZER_VERSION
    ):
        raise ValueError("Only the lossless board input format is supported")
    return ARCHITECTURES[architecture]


def create_model(config):
    arguments = dict(config)
    arguments.pop("architecture", None)
    arguments.pop("tokenizer_version", None)
    return model_class(config)(**arguments)
