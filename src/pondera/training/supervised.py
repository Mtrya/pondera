"""Policy-only supervised training on pre-encoded position arrays."""

import json
import math
import time
import tomllib
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from pondera.data.positions import TokenizedPositions
from pondera.models.checkpoint import read_checkpoint, save_checkpoint
from pondera.models.transformer import PonderaModel


@dataclass
class TrainConfig:
    model: dict = field(
        default_factory=lambda: {
            "num_blocks": 20,
            "hidden_size": 640,
            "intermediate_size": 1728,
            "num_heads": 8,
            "dropout": 0.05,
        }
    )
    seed: int = 640
    batch_size: int = 128
    accumulation_steps: int = 8
    epochs: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    grad_clip: float = 1.0
    workers: int = 4
    precision: str = "bf16"
    val_every: int = 500
    val_positions: int = 2048
    log_every: int = 50
    save_every: int = 2000

    def __post_init__(self):
        for name in (
            "batch_size",
            "accumulation_steps",
            "epochs",
            "val_every",
            "val_positions",
            "log_every",
            "save_every",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.workers < 0 or self.precision not in {"fp32", "bf16"}:
            raise ValueError("Invalid workers or precision")
        for name in ("learning_rate", "grad_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be between 0 and 1")


def load_config(path):
    with Path(path).open("rb") as handle:
        return TrainConfig(**tomllib.load(handle))


def train(
    config, data_dir, output_dir, device="cpu", resume=None, max_steps=None, track=False
):
    """Train or resume up to an absolute optimizer step without changing the schedule."""
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    device = torch.device(device)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and resume is None:
        raise FileExistsError(f"Use a new run directory or resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    train_data = TokenizedPositions(data_dir, "train")
    val_data = TokenizedPositions(data_dir, "val")
    micro_batches = len(train_data) // config.batch_size
    steps_per_epoch = micro_batches // config.accumulation_steps
    if not steps_per_epoch or not len(val_data):
        raise ValueError(
            "Training needs one full optimizer batch and nonempty validation data"
        )
    total_steps = steps_per_epoch * config.epochs
    model = PonderaModel(**config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * config.warmup_ratio), total_steps
    )
    step, start_epoch, next_micro = 0, 0, 0
    tracking_id = None
    checkpoint = None
    if resume:
        checkpoint = read_checkpoint(resume)
        if checkpoint["train_config"] != asdict(config):
            raise ValueError("Resume configuration differs from checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        step, start_epoch, next_micro = (
            checkpoint["step"],
            checkpoint["epoch"],
            checkpoint["next_micro"],
        )
        tracking_id = checkpoint.get("tracking_id")
    tracker = None
    if track:
        import swanlab

        tracker = swanlab
        run = tracker.init(
            project="pondera",
            experiment_name=output_dir.name,
            config=asdict(config),
            logdir=str(output_dir / "tracking"),
            id=tracking_id,
            resume="allow",
        )
        tracking_id = run.id
    if checkpoint:
        torch.set_rng_state(checkpoint["rng_state"])
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    def autocast():
        return (
            torch.autocast(device.type, dtype=torch.bfloat16)
            if config.precision == "bf16"
            else nullcontext()
        )

    @torch.no_grad()
    def validate():
        model.eval()
        correct, loss_sum, count = 0, 0.0, 0
        limit = min(config.val_positions, len(val_data))
        with autocast():
            for start in range(0, limit, config.batch_size):
                end = min(start + config.batch_size, limit)
                ids = torch.tensor(val_data.tokens[start:end], device=device)
                targets = torch.tensor(
                    val_data.moves[start:end], dtype=torch.long, device=device
                )
                logits, _ = model(ids=ids)
                loss_sum += torch.nn.functional.cross_entropy(
                    logits.float(), targets, reduction="sum"
                ).item()
                correct += (logits.argmax(-1) == targets).sum().item()
                count += len(targets)
        model.train()
        return {"val/top1": correct / count, "val/loss": loss_sum / count}

    def save(name, epoch, micro):
        save_checkpoint(
            output_dir / name,
            model,
            config.model,
            train_config=asdict(config),
            step=step,
            epoch=epoch,
            next_micro=micro,
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state_all()
            if device.type == "cuda"
            else None,
            tracking_id=tracking_id,
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    val_metrics = {}
    start_time = time.monotonic()
    initial_step = step
    epoch, cursor = start_epoch, next_micro
    limit = total_steps if max_steps is None else min(max_steps, total_steps)
    with (output_dir / "metrics.jsonl").open("a") as metrics:
        for epoch in range(start_epoch, config.epochs):
            if step >= limit:
                break
            loader = DataLoader(
                train_data,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.workers,
                drop_last=True,
                pin_memory=device.type == "cuda",
                generator=torch.Generator().manual_seed(config.seed + epoch),
                **(
                    {"persistent_workers": True, "prefetch_factor": 4}
                    if config.workers
                    else {}
                ),
            )
            loss_sum = 0.0
            for micro, (ids, targets) in enumerate(loader):
                if micro >= steps_per_epoch * config.accumulation_steps:
                    break
                if epoch == start_epoch and micro < next_micro:
                    continue
                with autocast():
                    logits, _ = model(ids=ids.to(device, non_blocking=True))
                    loss = torch.nn.functional.cross_entropy(
                        logits.float(), targets.to(device, non_blocking=True)
                    )
                (loss / config.accumulation_steps).backward()
                loss_sum += loss.item()
                if (micro + 1) % config.accumulation_steps:
                    continue
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                cursor = micro + 1
                if step == 1 or step % config.val_every == 0:
                    val_metrics = validate()
                if step == 1 or step % config.log_every == 0:
                    values = {
                        "step": step,
                        "epoch": epoch,
                        "train/loss": loss_sum / config.accumulation_steps,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/grad_norm": grad_norm.item(),
                        "train/pos_per_s": (step - initial_step)
                        * config.batch_size
                        * config.accumulation_steps
                        / (time.monotonic() - start_time),
                        **val_metrics,
                    }
                    metrics.write(json.dumps(values) + "\n")
                    metrics.flush()
                    print(json.dumps(values), flush=True)
                    if tracker:
                        tracker.log(values, step=step)
                loss_sum = 0.0
                if step % config.save_every == 0:
                    save(f"step-{step:06d}.pth", epoch, cursor)
                if step >= limit:
                    break
            if step >= limit:
                break
            cursor = 0
    save("final.pth", epoch, cursor)
    if tracker:
        tracker.finish()
    return output_dir / "final.pth"
