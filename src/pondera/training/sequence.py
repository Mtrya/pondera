"""Causal sequence training over encoded teacher traces.

Follows the depth.py training skeleton (resume, schedule, tracking) under the
sequence contract: the board prefix is attended bidirectionally, the trace
suffix causally, and supervision is shifted next-token CE on trace tokens only.
Line segments are reordered per presentation so candidate order carries no
teacher ranking; the FINAL segment always stays last. This trainer
intentionally duplicates the depth.py boilerplate instead of sharing it: the
two data and model contracts evolve independently.
"""

import argparse
import hashlib
import json
import math
import random
import time
import tomllib
import zlib
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from pondera.chess.vocabulary import PAD
from pondera.data.positions import SequencePositions
from pondera.models.checkpoint import read_checkpoint, save_checkpoint
from pondera.models.factory import create_model


@dataclass
class SequenceTrainConfig:
    model: dict = field(
        default_factory=lambda: {
            "architecture": "sequence",
            "num_blocks": 12,
            "hidden_size": 768,
            "intermediate_size": 2027,
            "num_heads": 12,
            "dropout": 0.0,
        }
    )
    seed: int = 640
    batch_size: int = 512
    accumulation_steps: int = 1
    epochs: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    grad_clip: float = 1.0
    workers: int = 4
    precision: str = "bf16"
    val_every: int = 128
    val_positions: int = 8192
    log_every: int = 50
    save_every: int = 1024

    def __post_init__(self):
        if self.model.get("architecture") != "sequence":
            raise ValueError("Sequence training requires a sequence-architecture model")
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


@dataclass
class SequenceCollator:
    """Pack boards, pad traces, and reorder LINE segments per presentation.

    The FINAL segment always stays last; only the leading LINE segments move,
    and shuffling changes presentation order alone, never token content. Every
    permutation is derived from the row's stored trace and the salt, not from
    the consumption order, so runs, worker processes, and resumed runs present
    the same row in the same order. With shuffle off the presented trace is
    exactly the stored one, the deterministic layout used for validation.
    """

    shuffle: bool = True
    salt: int = 0

    def __call__(self, rows):
        inputs = {"ids": torch.from_numpy(np.stack([row[0] for row in rows]))}
        traces, lengths, final_moves = [], [], []
        for _, trace, trace_len, line_bounds in rows:
            segments = [trace[start:end] for start, end in line_bounds if start >= 0]
            if not segments:
                raise ValueError("A trace row carries no segment bounds")
            if self.shuffle and len(segments) > 1:
                order = list(range(len(segments) - 1))
                random.Random(zlib.crc32(trace.tobytes(), self.salt)).shuffle(order)
                segments = [segments[index] for index in order] + [segments[-1]]
            presented = np.concatenate(segments)
            if len(presented) != trace_len:
                raise ValueError("Segment bounds do not tile the stored trace")
            traces.append(presented)
            lengths.append(trace_len)
            final_moves.append(sum(len(segment) for segment in segments[:-1]) + 1)
        width = max(lengths)
        trace_ids = np.full((len(traces), width), PAD, dtype=np.int16)
        for index, presented in enumerate(traces):
            trace_ids[index, : len(presented)] = presented
        inputs["trace_ids"] = torch.from_numpy(trace_ids)
        inputs["final_move_index"] = torch.tensor(final_moves, dtype=torch.long)
        return inputs


def load_config(path):
    with Path(path).open("rb") as handle:
        return SequenceTrainConfig(**tomllib.load(handle))


def trace_loss(model, inputs):
    """Masked mean next-token CE over the trace, plus the target positions."""
    trace_ids = inputs["trace_ids"]
    targets = trace_ids.long()
    logits = model.trace_logits(inputs["ids"], trace_ids)
    mask = trace_ids != PAD
    ce = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape(trace_ids.shape)
    return (ce * mask).sum() / mask.sum().clamp_min(1), logits, mask


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
    train_data = SequencePositions(data_dir, "train")
    val_data = SequencePositions(data_dir, "val")
    eval_collate = SequenceCollator(shuffle=False)
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=True if device.type == "cuda" else None,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * config.warmup_ratio), total_steps
    )
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
        execution_history = list(checkpoint.get("execution_history", execution_history))
        if previous_config != current_config:
            execution_history.append(
                {
                    "step": step,
                    "batch_size": config.batch_size,
                    "accumulation_steps": config.accumulation_steps,
                }
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
    def evaluate(data, indices, prefix):
        model.eval()
        loss_sum = torch.zeros((), device=device)
        token_count, correct, count = 0, 0, 0
        for start in range(0, len(indices), config.batch_size):
            selected = indices[start : start + config.batch_size]
            inputs = {
                key: value.to(device)
                for key, value in eval_collate([data[i] for i in selected]).items()
            }
            with autocast():
                loss, logits, mask = trace_loss(model, inputs)
            loss_sum += float(loss) * int(mask.sum())
            token_count += int(mask.sum())
            final = inputs["final_move_index"]
            selected_logits = logits.gather(
                1, final[:, None, None].expand(-1, 1, logits.shape[-1])
            ).squeeze(1)
            correct += int(
                (
                    selected_logits.argmax(-1)
                    == inputs["trace_ids"].gather(1, final[:, None]).squeeze(1)
                ).sum()
            )
            count += len(selected)
        model.train()
        return {
            f"{prefix}/loss": loss_sum.item() / max(token_count, 1),
            f"{prefix}/final_move_top1": correct / count,
            f"{prefix}/rows": count,
        }

    examples_seen = step * config.batch_size * config.accumulation_steps

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
            if step >= limit:
                break
            train_collate = SequenceCollator(shuffle=True, salt=config.seed + epoch)
            loader = DataLoader(
                train_data,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.workers,
                drop_last=True,
                pin_memory=device.type == "cuda",
                generator=torch.Generator().manual_seed(config.seed + epoch),
                collate_fn=train_collate,
                **(
                    {"persistent_workers": True, "prefetch_factor": 4}
                    if config.workers
                    else {}
                ),
            )
            loss_sum = torch.zeros((), device=device)
            for micro, inputs in enumerate(loader):
                if micro >= steps_per_epoch * config.accumulation_steps:
                    break
                if epoch == start_epoch and micro < next_micro:
                    continue
                examples_seen += config.batch_size
                inputs = {
                    key: value.to(device, non_blocking=True)
                    for key, value in inputs.items()
                }
                with autocast():
                    loss, _, _ = trace_loss(model, inputs)
                (loss / config.accumulation_steps).backward()
                loss_sum += loss.detach()
                if (micro + 1) % config.accumulation_steps:
                    continue
                token_grad = model.token_embedding.weight.grad
                token_grad_norm = (
                    None if token_grad is None else token_grad.norm().item()
                )
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
                    val_metrics = evaluate(
                        val_data, range(min(config.val_positions, len(val_data))), "val"
                    )
                if should_validate or step % config.log_every == 0:
                    values = {
                        "step": step,
                        "epoch": epoch,
                        "train/loss": (loss_sum / config.accumulation_steps).item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/grad_norm": grad_norm.item(),
                        "train/embedding_grad_norm": token_grad_norm,
                        "train/rows_per_s": (step - initial_step)
                        * config.batch_size
                        * config.accumulation_steps
                        / (time.monotonic() - start_time),
                        "train/examples": examples_seen,
                        "train/data_passes": examples_seen / len(train_data),
                        "train/schedule_progress": step / total_steps,
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
                if step >= limit:
                    break
            if step >= limit:
                break
            cursor = 0
    final_metrics = {
        "step": step,
        "final": True,
        "training_complete": step >= total_steps,
        "train/examples": examples_seen,
        "train/data_passes": examples_seen / len(train_data),
        **evaluate(val_data, range(min(config.val_positions, len(val_data))), "val"),
    }
    with (output_dir / "metrics.jsonl").open("a") as metrics:
        metrics.write(json.dumps(final_metrics) + "\n")
    print(json.dumps(final_metrics), flush=True)
    save("final.pth", epoch, cursor)
    if tracker:
        tracker.finish()
    return output_dir / "final.pth"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Stop at this absolute optimizer step, including steps before resume",
    )
    parser.add_argument("--track", action="store_true", help="Log metrics to SwanLab")
    args = parser.parse_args()
    print(
        train(
            load_config(args.config),
            args.data_dir,
            args.output_dir,
            args.device,
            args.resume,
            args.max_steps,
            args.track,
        )
    )


if __name__ == "__main__":
    main()
