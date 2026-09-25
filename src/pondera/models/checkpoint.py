"""Model checkpoint I/O shared by training, inference, and evaluation."""

import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from pondera.models.factory import create_model, model_tokenizer_version
from pondera.models.policy import BoardPolicyModel, DepthPolicyModel
from pondera.models.transformer import PonderaModel

CHECKPOINT_VERSION = 1


def read_checkpoint(path, device="cpu") -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    version = checkpoint.get("format_version", CHECKPOINT_VERSION)
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {version}")
    # Unversioned baseline checkpoints use the original token layout (version 1).
    tokenizer_version = checkpoint.get(
        "tokenizer_version", 1 if "format_version" not in checkpoint else None
    )
    # Existing published baselines used either of these configuration keys.
    if "model_config" not in checkpoint:
        checkpoint["model_config"] = checkpoint["config"]
    if tokenizer_version != model_tokenizer_version(checkpoint["model_config"]):
        raise ValueError(
            f"Unsupported checkpoint tokenizer version: {tokenizer_version}"
        )
    return checkpoint


def save_checkpoint(path, model, model_config, **training_state) -> None:
    if model.tokenizer_version != model_tokenizer_version(model_config):
        raise ValueError(
            "Checkpoint configuration disagrees with the model's input format"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        **training_state,
        "format_version": CHECKPOINT_VERSION,
        "tokenizer_version": model_tokenizer_version(model_config),
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_model(source, device="cpu") -> torch.nn.Module:
    """Load a checkpoint file or Hugging Face model directory/repository."""
    path = Path(source)
    if path.is_file():
        checkpoint = read_checkpoint(path)
        model = create_model(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
    elif path.is_dir():
        config = json.loads((path / "config.json").read_text())
        model_tokenizer_version(config)
        classes = {
            "legacy": PonderaModel,
            "base": BoardPolicyModel,
            "depth": DepthPolicyModel,
        }
        model = classes[config.get("architecture", "legacy")].from_pretrained(str(path))
    elif path.suffix in {".pth", ".pt"} or isinstance(source, Path):
        if not path.exists():
            raise FileNotFoundError(path)
        model = PonderaModel.from_pretrained(str(path))
    else:
        config_path = hf_hub_download(str(source), "config.json")
        config = json.loads(Path(config_path).read_text())
        model_tokenizer_version(config)
        classes = {
            "legacy": PonderaModel,
            "base": BoardPolicyModel,
            "depth": DepthPolicyModel,
        }
        model = classes[config.get("architecture", "legacy")].from_pretrained(
            str(source)
        )
    return model.to(device).eval()
