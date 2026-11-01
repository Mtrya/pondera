"""Model checkpoint I/O shared by training, inference, and evaluation."""

import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from pondera.chess.vocabulary import CHESS_TOKENIZER_VERSION
from pondera.models.policy import PolicyModel, create_model

CHECKPOINT_VERSION = 1


def read_checkpoint(path, device="cpu") -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format version: {checkpoint.get('format_version')}"
        )
    if checkpoint.get("tokenizer_version") != CHESS_TOKENIZER_VERSION:
        raise ValueError(
            "Unsupported checkpoint input format version: "
            f"{checkpoint.get('tokenizer_version')}"
        )
    if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing its model configuration or weights")
    return checkpoint


def save_checkpoint(path, model, model_config, **training_state) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        **training_state,
        "format_version": CHECKPOINT_VERSION,
        "tokenizer_version": CHESS_TOKENIZER_VERSION,
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_model(source, device="cpu") -> torch.nn.Module:
    path = Path(source)
    if path.is_file():
        checkpoint = read_checkpoint(path)
        model = create_model(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(device).eval()
    if path.is_dir():
        repository, config_path = str(path), path / "config.json"
    else:
        repository = str(source)
        config_path = Path(hf_hub_download(repository, "config.json"))
    config = json.loads(config_path.read_text())
    if (
        config.get("tokenizer_version", CHESS_TOKENIZER_VERSION)
        != CHESS_TOKENIZER_VERSION
    ):
        raise ValueError("Only the lossless board input format is supported")
    model = PolicyModel.from_pretrained(repository)
    return model.to(device).eval()
