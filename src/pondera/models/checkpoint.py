"""Model checkpoint I/O shared by training, inference, and evaluation."""

import os
from pathlib import Path

import torch

from pondera.models.transformer import PonderaModel

CHECKPOINT_VERSION = 1


def read_checkpoint(path, device="cpu") -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    version = checkpoint.get("format_version", CHECKPOINT_VERSION)
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {version}")
    # Existing published baselines used either of these configuration keys.
    if "model_config" not in checkpoint:
        checkpoint["model_config"] = checkpoint["config"]
    return checkpoint


def save_checkpoint(path, model, model_config, **training_state) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        **training_state,
        "format_version": CHECKPOINT_VERSION,
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_model(source, device="cpu") -> PonderaModel:
    """Load a checkpoint file or Hugging Face model directory/repository."""
    path = Path(source)
    if path.is_file():
        checkpoint = read_checkpoint(path)
        model = PonderaModel(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
    elif path.suffix in {".pth", ".pt"} or isinstance(source, Path):
        if not path.exists():
            raise FileNotFoundError(path)
        model = PonderaModel.from_pretrained(str(path))
    else:
        model = PonderaModel.from_pretrained(str(source))
    return model.to(device).eval()
