"""Explicit model selection for serialized policy configurations."""

from pondera.models.policy import BoardPolicyModel, DepthPolicyModel
from pondera.models.transformer import PonderaModel


def model_tokenizer_version(config):
    architecture = config.get("architecture", "legacy")
    if architecture not in {"legacy", "base", "depth"}:
        raise ValueError(f"Unknown policy architecture: {architecture}")
    expected = 1 if architecture == "legacy" else 2
    if config.get("tokenizer_version", expected) != expected:
        raise ValueError("Model architecture and tokenizer version disagree")
    return expected


def create_model(config):
    model_tokenizer_version(config)
    arguments = dict(config)
    architecture = arguments.pop("architecture", "legacy")
    constructors = {
        "legacy": PonderaModel,
        "base": BoardPolicyModel,
        "depth": DepthPolicyModel,
    }
    return constructors[architecture](**arguments)
