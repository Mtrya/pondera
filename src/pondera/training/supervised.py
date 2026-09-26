"""Policy-only supervised training on pre-encoded position arrays."""

import hashlib
import json
import math
import time
import tomllib
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from pondera.chess.board_tokens import PAD
from pondera.chess.mapping import EMPTY_SQ_IDX
from pondera.data.positions import TokenizedPositions
from pondera.models.board import pack_board_tokens
from pondera.models.checkpoint import read_checkpoint, save_checkpoint
from pondera.models.factory import create_model, model_tokenizer_version
from pondera.training.cost import policy_flops
from pondera.training.metrics import PolicyMetrics, policy_loss, readout_changes


@dataclass
class TrainConfig:
    model: dict = field(
        default_factory=lambda: {
            "architecture": "base",
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
    train_flops_budget: float | None = None
    diagnostics: bool = False
    legal_loss: bool = False
    train_probe_positions: int = 0

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
        if self.train_probe_positions < 0:
            raise ValueError("train_probe_positions must be non-negative")
        if (self.diagnostics or self.legal_loss) and self.model.get(
            "architecture"
        ) != "base":
            raise ValueError(
                "Policy diagnostics and legal loss require a base board policy"
            )
        for name in ("learning_rate", "grad_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be between 0 and 1")
        if self.train_flops_budget is not None:
            if (
                not math.isfinite(self.train_flops_budget)
                or self.train_flops_budget <= 0
            ):
                raise ValueError("train_flops_budget must be finite and positive")
            if self.model.get("architecture") != "base":
                raise ValueError(
                    "FLOP-budgeted training currently requires a base board policy"
                )


@dataclass
class PolicyCollator:
    layout: str | None = None

    def __call__(self, rows):
        ids = torch.from_numpy(np.stack([row[0] for row in rows]))
        inputs = (
            {"ids": ids} if self.layout is None else pack_board_tokens(ids, self.layout)
        )
        targets = torch.tensor([row[1] for row in rows], dtype=torch.long)
        metadata = {}
        if len(rows[0]) == 3:
            metadata = {
                "legal": torch.from_numpy(np.stack([row[2] for row in rows])).long(),
                "pieces": (ids[:, :64] != EMPTY_SQ_IDX).sum(-1),
            }
        return inputs, targets, metadata


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
    tokenizer_version = model_tokenizer_version(config.model)
    need_legal = config.diagnostics or config.legal_loss
    train_data = TokenizedPositions(
        data_dir, "train", tokenizer_version, legal=need_legal
    )
    val_data = TokenizedPositions(data_dir, "val", tokenizer_version, legal=need_legal)
    collate = PolicyCollator(
        config.model.get("layout", "squares") if tokenizer_version == 2 else None
    )
    data_identity = hashlib.sha256(
        json.dumps(train_data.manifest, sort_keys=True).encode()
    ).hexdigest()
    micro_batches = len(train_data) // config.batch_size
    steps_per_epoch = micro_batches // config.accumulation_steps
    if not steps_per_epoch or not len(val_data):
        raise ValueError(
            "Training needs one full optimizer batch and nonempty validation data"
        )
    total_steps = steps_per_epoch * config.epochs
    model = create_model(config.model).to(device)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    consumed_flops = 0.0
    examples_seen = 0
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=True if device.type == "cuda" else None,
    )
    if config.train_flops_budget is None:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, int(total_steps * config.warmup_ratio), total_steps
        )
    else:

        def learning_rate_fraction(_):
            progress = min(1.0, consumed_flops / config.train_flops_budget)
            if progress < config.warmup_ratio:
                return progress / config.warmup_ratio
            decay = (progress - config.warmup_ratio) / max(
                1e-12, 1 - config.warmup_ratio
            )
            return 0.5 * (1 + math.cos(math.pi * decay))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_fraction)
    step, start_epoch, next_micro = 0, 0, 0
    tracking_id = None
    checkpoint = None
    execution_history = [
        {
            "step": 0,
            "batch_size": config.batch_size,
            "accumulation_steps": config.accumulation_steps,
        }
    ]
    if resume:
        checkpoint = read_checkpoint(resume)
        previous_config = checkpoint["train_config"]
        current_config = asdict(config)
        batching_keys = {"batch_size", "accumulation_steps"}
        previous_effective = (
            previous_config["batch_size"] * previous_config["accumulation_steps"]
        )
        current_effective = config.batch_size * config.accumulation_steps
        if previous_effective != current_effective or (
            {
                key: value
                for key, value in previous_config.items()
                if key not in batching_keys
            }
            != {
                key: value
                for key, value in current_config.items()
                if key not in batching_keys
            }
        ):
            raise ValueError("Resume configuration differs from checkpoint")
        if checkpoint.get("data_identity", data_identity) != data_identity:
            raise ValueError("Resume dataset differs from checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        step, start_epoch, next_micro = (
            checkpoint["step"],
            checkpoint["epoch"],
            checkpoint["next_micro"],
        )
        consumed_in_epoch = next_micro * previous_config["batch_size"]
        if consumed_in_epoch % config.batch_size:
            raise ValueError("Checkpoint is not aligned with the requested microbatch")
        next_micro = consumed_in_epoch // config.batch_size
        execution_history = list(
            checkpoint.get(
                "execution_history",
                [
                    {
                        "step": 0,
                        "batch_size": previous_config["batch_size"],
                        "accumulation_steps": previous_config["accumulation_steps"],
                    }
                ],
            )
        )
        if previous_config != current_config:
            execution_history.append(
                {
                    "step": step,
                    "batch_size": config.batch_size,
                    "accumulation_steps": config.accumulation_steps,
                }
            )
        tracking_id = checkpoint.get("tracking_id")
        consumed_flops = checkpoint.get("train_flops", 0.0)
        examples_seen = checkpoint.get(
            "examples_seen", step * config.batch_size * config.accumulation_steps
        )
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
    (output_dir / "execution.json").write_text(
        json.dumps(execution_history, indent=2) + "\n"
    )

    def autocast():
        return (
            torch.autocast(device.type, dtype=torch.bfloat16)
            if config.precision == "bf16"
            else nullcontext()
        )

    @torch.no_grad()
    def evaluate(data, indices, prefix, artifact=None):
        model.eval()
        correct = torch.zeros((), dtype=torch.long, device=device)
        loss_sum = torch.zeros((), device=device)
        count = 0
        observations = PolicyMetrics() if config.diagnostics else None
        internal = {}
        with autocast():
            for start in range(0, len(indices), config.batch_size):
                selected = indices[start : start + config.batch_size]
                inputs, targets, metadata = collate([data[i] for i in selected])
                targets = targets.to(device)
                inputs = {key: value.to(device) for key, value in inputs.items()}
                if observations is not None and start == 0 and prefix == "val":
                    logits, _, output = model(
                        **inputs, return_backbone=True, capture=True
                    )
                    internal = readout_changes(output)
                    del output
                else:
                    logits, _ = model(**inputs)
                loss_sum += torch.nn.functional.cross_entropy(
                    logits.float(), targets, reduction="sum"
                )
                correct += (logits.argmax(-1) == targets).sum()
                count += len(targets)
                if observations is not None:
                    observations.update(
                        logits,
                        targets,
                        metadata["legal"].to(device),
                        metadata["pieces"].to(device),
                    )
        model.train()
        values = {"top1": correct.item() / count, "loss": loss_sum.item() / count}
        if observations is not None:
            values.update(observations.result(artifact))
        return {
            **{f"{prefix}/{key}": value for key, value in values.items()},
            **{f"internal/{key}": value for key, value in internal.items()},
        }

    def validate():
        values = evaluate(
            val_data,
            range(min(config.val_positions, len(val_data))),
            "val",
            output_dir / "probes" / f"val-{step:06d}.npz",
        )
        if config.train_probe_positions:
            indices = np.linspace(
                0,
                len(train_data) - 1,
                min(config.train_probe_positions, len(train_data)),
                dtype=int,
            )
            values.update(evaluate(train_data, indices, "train_probe"))
            values["generalization/loss_gap"] = (
                values["val/loss"] - values["train_probe/loss"]
            )
            if config.diagnostics:
                values["generalization/legal_loss_gap"] = (
                    values["val/legal_loss"] - values["train_probe/legal_loss"]
                )
        return values

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
            train_flops=consumed_flops,
            examples_seen=examples_seen,
            data_identity=data_identity,
            execution_history=execution_history,
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    val_metrics = {}
    start_time = time.monotonic()
    initial_step = step
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    epoch, cursor = start_epoch, next_micro
    limit = total_steps if max_steps is None else min(max_steps, total_steps)
    with (output_dir / "metrics.jsonl").open("a") as metrics:
        for epoch in range(start_epoch, config.epochs):
            if step >= limit or (
                config.train_flops_budget is not None
                and consumed_flops >= config.train_flops_budget
            ):
                break
            loader = DataLoader(
                train_data,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.workers,
                drop_last=True,
                pin_memory=device.type == "cuda",
                generator=torch.Generator().manual_seed(config.seed + epoch),
                collate_fn=collate,
                **(
                    {"persistent_workers": True, "prefetch_factor": 4}
                    if config.workers
                    else {}
                ),
            )
            loss_sum = torch.zeros((), device=device)
            for micro, (inputs, targets, metadata) in enumerate(loader):
                if micro >= steps_per_epoch * config.accumulation_steps:
                    break
                if epoch == start_epoch and micro < next_micro:
                    continue
                if config.model.get("architecture") == "base":
                    lengths = (inputs["ids"] != PAD).sum(1).numpy()
                    consumed_flops += float(policy_flops(config.model, lengths).sum())
                examples_seen += len(targets)
                with autocast():
                    logits, _ = model(
                        **{
                            key: value.to(device, non_blocking=True)
                            for key, value in inputs.items()
                        }
                    )
                    loss = policy_loss(
                        logits.float(),
                        targets.to(device, non_blocking=True),
                        metadata["legal"].to(device, non_blocking=True)
                        if config.legal_loss
                        else None,
                    )
                (loss / config.accumulation_steps).backward()
                loss_sum += loss.detach()
                if (micro + 1) % config.accumulation_steps:
                    continue
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                cursor = micro + 1
                val_metrics = {}
                should_validate = step == 1 or step % config.val_every == 0
                if should_validate:
                    val_metrics = validate()
                if should_validate or step % config.log_every == 0:
                    values = {
                        "step": step,
                        "epoch": epoch,
                        "train/loss": (loss_sum / config.accumulation_steps).item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/grad_norm": grad_norm.item(),
                        "train/pos_per_s": (step - initial_step)
                        * config.batch_size
                        * config.accumulation_steps
                        / (time.monotonic() - start_time),
                        "train/theoretical_flops": consumed_flops
                        if config.model.get("architecture") == "base"
                        else None,
                        "train/examples": examples_seen,
                        "train/data_passes": examples_seen / len(train_data),
                        "train/schedule_progress": min(
                            1.0, consumed_flops / config.train_flops_budget
                        )
                        if config.train_flops_budget
                        else step / total_steps,
                        "model/parameters": trainable_parameters,
                        "elapsed_seconds": time.monotonic() - start_time,
                        "memory/peak_allocated_bytes": torch.cuda.max_memory_allocated(
                            device
                        )
                        if device.type == "cuda"
                        else None,
                        **val_metrics,
                    }
                    metrics.write(json.dumps(values) + "\n")
                    metrics.flush()
                    print(json.dumps(values), flush=True)
                    if tracker:
                        tracker.log(values, step=step)
                loss_sum.zero_()
                if step % config.save_every == 0:
                    save(f"step-{step:06d}.pth", epoch, cursor)
                if step >= limit or (
                    config.train_flops_budget is not None
                    and consumed_flops >= config.train_flops_budget
                ):
                    break
            if step >= limit or (
                config.train_flops_budget is not None
                and consumed_flops >= config.train_flops_budget
            ):
                break
            cursor = 0
    final_metrics = {
        "step": step,
        "final": True,
        "training_complete": step >= total_steps
        or (
            config.train_flops_budget is not None
            and consumed_flops >= config.train_flops_budget
        ),
        "train/theoretical_flops": consumed_flops
        if config.model.get("architecture") == "base"
        else None,
        "train/examples": examples_seen,
        "flop_budget_reached": config.train_flops_budget is not None
        and consumed_flops >= config.train_flops_budget,
        **validate(),
    }
    with (output_dir / "metrics.jsonl").open("a") as metrics:
        metrics.write(json.dumps(final_metrics) + "\n")
    print(json.dumps(final_metrics), flush=True)
    save("final.pth", epoch, cursor)
    if tracker:
        tracker.finish()
    return output_dir / "final.pth"
