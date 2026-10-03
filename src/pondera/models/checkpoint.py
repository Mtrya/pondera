"""Model checkpoint I/O shared by training, inference, and evaluation."""

import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from pondera.chess.tokens import CHESS_TOKENIZER_VERSION
from pondera.models.factory import create_model, model_class

CHECKPOINT_VERSION = 1

# Vocabulary rows a checkpoint never trained are masked out of selection by a
# strongly negative head bias, so their logits cannot win a softmax or argmax.
MASKED_BIAS = -1e4


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


def adapt_state_dict(model, state):
    """Fit checkpoint tensors to the model's shapes along the vocabulary axis.

    The unified vocabulary is append-only, so a checkpoint saved under an
    earlier one keeps exactly the outputs it was trained on: vocabulary rows
    the model grew are zero, except a head bias, which is filled with a
    strongly negative value so the added ids are never selected. Rows the
    model dropped are truncated. Any other shape mismatch is an error.
    """
    expected = model.state_dict()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    if missing or unexpected:
        raise ValueError(
            "Checkpoint keys do not match the model: "
            f"missing {sorted(missing)}, unexpected {sorted(unexpected)}"
        )
    adapted = {}
    for key, target in expected.items():
        tensor = state[key]
        if tensor.shape == target.shape:
            adapted[key] = tensor
            continue
        if tensor.ndim != target.ndim or tensor.shape[1:] != target.shape[1:]:
            raise ValueError(
                f"Checkpoint tensor {key} has shape {tuple(tensor.shape)}, "
                f"the model expects {tuple(target.shape)}"
            )
        fill = MASKED_BIAS if key.endswith("bias") else 0.0
        fitted = torch.full(
            target.shape, fill, dtype=target.dtype, device=tensor.device
        )
        rows = min(tensor.shape[0], target.shape[0])
        fitted[:rows] = tensor[:rows].to(target.dtype)
        adapted[key] = fitted
    return adapted


def load_model(source, device="cpu") -> torch.nn.Module:
    """Load a checkpoint file or a Hugging Face model directory/repository."""
    path = Path(source)
    if path.is_file():
        checkpoint = read_checkpoint(path)
        model = create_model(checkpoint["model_config"])
        model.load_state_dict(adapt_state_dict(model, checkpoint["model_state_dict"]))
        return model.to(device).eval()
    if path.is_dir():
        repository, config_path = str(path), path / "config.json"
    else:
        repository = str(source)
        config_path = Path(hf_hub_download(repository, "config.json"))
    model = model_class(json.loads(config_path.read_text())).from_pretrained(repository)
    return model.to(device).eval()
