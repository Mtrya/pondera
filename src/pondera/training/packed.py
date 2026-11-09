"""Minimal trainer for the frozen packed batch contract.

Muon owns hidden matrices, AdamW the rest (training/muon.py). One optimizer
update consumes whole microbatches until its token target is met; each
microbatch's equal-weight next-token CE is normalized by that microbatch's
labeled positions and weighted by its share of the update's actual labeled
tokens. The batch sequence is a pure function of (subset, seed, budget,
step), so resume replays it exactly from the saved micro step.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pondera.data.packing import (
    PackedDocuments,
    assemble_batch,
    load_subsets,
    pack_batches,
    packed_batches,
)
from pondera.training.muon import build_optimizers


@dataclass
class TrainConfig:
    muon_lr: float = 0.02
    adamw_lr: float = 3e-4
    weight_decay: float = 0.0
    microbatch_tokens: int = 65_536
    update_tokens: int = 524_288
    updates: int = 100
    val_every: int = 10
    checkpoint_every: int = 50
    seed: int = 640
    track: bool = False


def _move(batch, device):
    dtypes = {"input_ids": torch.long, "labels": torch.long, "cu_seqlens": torch.int32}
    return {
        key: torch.as_tensor(value, dtype=dtypes[key], device=device)
        for key, value in batch.items()
        if key in dtypes
    }


def _loss(model, batch):
    device_type = model.token_embedding.weight.device.type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=device_type in {"cuda", "npu"},
    ):
        logits = model(batch["input_ids"], batch["cu_seqlens"])
    labeled = int((batch["labels"] != -100).sum())
    loss = F.cross_entropy(
        logits.float().view(-1, logits.shape[-1]),
        batch["labels"].view(-1),
        ignore_index=-100,
    )
    return loss, labeled


@torch.no_grad()
def validation_loss(model, documents, budget, device):
    indices = np.flatnonzero(documents.split == 1)
    model.eval()
    batches, _ = pack_batches(documents.lengths, indices, budget)
    total_loss = total_labeled = 0
    for batch in batches:
        moved = _move(assemble_batch(documents, list(batch)), device)
        loss, labeled = _loss(model, moved)
        total_loss += float(loss) * labeled
        total_labeled += labeled
    model.train()
    return total_loss / max(total_labeled, 1)


def save_training_checkpoint(
    path, model, optimizers, model_config, micro_step, update, config, manifest
):
    payload = {
        "format_version": 1,
        "arch": model.architecture,
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dicts": [optimizer.state_dict() for optimizer in optimizers],
        "micro_step": micro_step,
        "update": update,
        "train_config": asdict(config),
        "data_identity": manifest.get("source", {}),
        "tokenizer_identity": manifest.get("vocabulary", {}),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(
    model,
    model_config,
    packed_dir,
    subset,
    config: TrainConfig,
    output_dir,
    resume=None,
    device="cpu",
):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and resume is None:
        raise FileExistsError(f"Use a new run directory or resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    documents = PackedDocuments(packed_dir)
    indices = load_subsets(packed_dir)[subset]
    optimizers = build_optimizers(
        model,
        muon_lr=config.muon_lr,
        adamw_lr=config.adamw_lr,
        weight_decay=config.weight_decay,
    )
    micro_step = update = 0
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        for optimizer, saved in zip(optimizers, state["optimizer_state_dicts"]):
            optimizer.load_state_dict(saved)
        micro_step, update = state["micro_step"], state["update"]
    model.to(device).train()
    tracker = None
    if config.track:
        import swanlab

        tracker = swanlab.init(
            project="pondera-model-arch",
            config={"subset": subset, "model": model_config, **asdict(config)},
            resume="allow",
        )
    (output_dir / "config.json").write_text(
        json.dumps(
            {"model": model_config, "train": asdict(config), "subset": subset}, indent=2
        )
        + "\n"
    )
    history = []
    batch_stream = packed_batches(
        documents,
        indices,
        config.seed,
        config.microbatch_tokens,
        start_step=micro_step,
    )
    started = time.monotonic()
    while update < config.updates:
        batches = []
        total_labeled = 0
        total_tokens = 0
        while total_tokens < config.update_tokens:
            batch = next(batch_stream)
            micro_step += 1
            moved = _move(batch, device)
            moved["epoch"] = batch.get("epoch", 0)
            labeled = int((moved["labels"] != -100).sum())
            batches.append((moved, labeled))
            total_labeled += labeled
            total_tokens += moved["input_ids"].numel()
        for optimizer in optimizers:
            optimizer.zero_grad()
        update_loss = 0.0
        for batch, labeled in batches:
            loss, _ = _loss(model, batch)
            (loss * (labeled / total_labeled)).backward()
            update_loss += float(loss.detach()) * labeled / total_labeled
        for optimizer in optimizers:
            optimizer.step()
        update += 1
        record = {
            "update": update,
            "micro_step": micro_step,
            "loss": update_loss,
            "tokens": total_tokens,
            "labeled_tokens": total_labeled,
            "epoch": batches[-1][0].get("epoch", 0),
        }
        if config.val_every and update % config.val_every == 0:
            record["val_loss"] = validation_loss(
                model, documents, config.microbatch_tokens, device
            )
        history.append(record)
        if tracker is not None:
            tracker.log(record, step=update)
        if config.checkpoint_every and update % config.checkpoint_every == 0:
            save_training_checkpoint(
                output_dir / "checkpoint.pth",
                model,
                optimizers,
                model_config,
                micro_step,
                update,
                config,
                documents.manifest,
            )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    save_training_checkpoint(
        output_dir / "checkpoint.pth",
        model,
        optimizers,
        model_config,
        micro_step,
        update,
        config,
        documents.manifest,
    )
    if tracker is not None:
        tracker.finish()
    return {
        "updates": update,
        "micro_steps": micro_step,
        "seconds": time.monotonic() - started,
        "history": history,
    }


def main():
    import argparse

    from pondera.models.linear import LinearVanillaModel
    from pondera.models.vanilla import VanillaModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed", required=True, type=Path)
    parser.add_argument("--subset", default="500k")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--architecture", choices=("vanilla", "linear"), default="vanilla"
    )
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=2816)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    for field in TrainConfig.__dataclass_fields__:
        default = getattr(TrainConfig, field)
        if isinstance(default, bool):
            parser.add_argument(
                f"--{field.replace('_', '-')}", action="store_true", default=default
            )
        else:
            parser.add_argument(
                f"--{field.replace('_', '-')}", type=type(default), default=default
            )
    args = parser.parse_args()
    model_config = dict(
        num_blocks=args.num_blocks,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        vocab_size=PackedDocuments(args.packed).manifest["vocabulary"],
    )
    config = TrainConfig(
        **{field: getattr(args, field) for field in TrainConfig.__dataclass_fields__}
    )
    architecture = args.architecture
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        model_config = state["model_config"]
        architecture = state["arch"]
    torch.manual_seed(config.seed)
    model_class = {"vanilla": VanillaModel, "linear": LinearVanillaModel}[architecture]
    model = model_class(**model_config)
    result = train(
        model,
        model_config,
        args.packed,
        args.subset,
        config,
        args.output,
        resume=args.resume,
        device=args.device,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "history"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
