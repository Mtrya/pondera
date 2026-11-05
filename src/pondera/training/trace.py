"""Resumable next-token training over encoded teacher traces."""

import argparse
import json
import math
import time
import tomllib
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from pondera.chess.vocabulary import (
    BOARD_VOCAB_SIZE,
    CHESS_VOCAB_SIZE,
    END,
    FINAL,
    PAD,
)
from pondera.data.dataset import TracePositions, trace_data_identity
from pondera.data.trace import (
    ROLE_BOARD,
    ROLE_CONTROL,
    ROLE_COUNT,
    ROLE_EVAL,
    ROLE_FINAL_EVAL,
    ROLE_FINAL_MOVE,
    ROLE_FINAL_PV,
    ROLE_MOVE,
    ROLE_PAD,
    ROLE_REFUTED,
)
from pondera.models.checkpoint import read_checkpoint, save_checkpoint
from pondera.models.policy import create_model


@dataclass
class TrainConfig:
    model: dict = field(
        default_factory=lambda: {
            "num_blocks": 12,
            "hidden_size": 768,
            "intermediate_size": 2048,
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
    train_probe_positions: int = 2048
    log_every: int = 50
    save_every: int = 1024

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
        if self.train_probe_positions < 0:
            raise ValueError("train_probe_positions must be non-negative")
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
class TraceCollator:
    """Pad traces and derived roles, and locate each row's final decision.

    The trace already carries its presentation order. The dataset derives each
    initial board from its FEN and repetition count; boards are padded to the
    batch maximum. Every row ends with ``<end>``, and exactly one FINAL
    delimiter is followed by the decision that carries ROLE_FINAL_MOVE. The
    label suffix continues after that decision, so the final move is not the
    last token.
    """

    def __call__(self, rows):
        boards = [row[0] for row in rows]
        board_width = max((len(board) for board in boards), default=0)
        board_ids = np.full((len(boards), board_width), PAD, dtype=np.int16)
        for index, board in enumerate(boards):
            invalid = (board != PAD) & ((board < 0) | (board >= BOARD_VOCAB_SIZE))
            if np.any(invalid):
                raise ValueError("A board row contains a non-board token")
            board_ids[index, : len(board)] = board
        inputs = {"ids": torch.from_numpy(board_ids)}
        traces, roles, final_moves = [], [], []
        for _, trace, trace_len, trace_roles in rows:
            trace_len = int(trace_len)
            if len(trace) != len(trace_roles) or not 4 <= trace_len <= len(trace):
                raise ValueError("A trace row has inconsistent token boundaries")
            stored = trace[:trace_len]
            if np.any((stored < 0) | (stored >= CHESS_VOCAB_SIZE)):
                raise ValueError("A trace row contains an out-of-vocabulary token")
            if stored[-1] != END or np.count_nonzero(stored == END) != 1:
                raise ValueError("A trace row must end with <end>")
            stored_roles = trace_roles[:trace_len]
            if np.any((stored_roles < 0) | (stored_roles >= ROLE_COUNT)):
                raise ValueError("A trace row contains an invalid token role")
            final = np.flatnonzero(stored_roles == ROLE_FINAL_MOVE)
            if len(final) != 1:
                raise ValueError("A trace row needs exactly one final move role")
            decision = int(final[0])
            if decision == 0 or stored[decision - 1] != FINAL:
                raise ValueError("The final move must follow FINAL")
            traces.append(stored)
            roles.append(stored_roles)
            final_moves.append(decision)
        width = max((len(trace) for trace in traces), default=0)
        trace_ids = np.full((len(traces), width), PAD, dtype=np.int16)
        role_ids = np.full((len(roles), width), ROLE_PAD, dtype=np.int8)
        for index, (trace, trace_roles) in enumerate(zip(traces, roles)):
            trace_ids[index, : len(trace)] = trace
            role_ids[index, : len(trace_roles)] = trace_roles
        inputs["trace_ids"] = torch.from_numpy(trace_ids)
        inputs["trace_roles"] = torch.from_numpy(role_ids)
        inputs["final_move_index"] = torch.tensor(final_moves, dtype=torch.long)
        return inputs


def load_config(path):
    with Path(path).open("rb") as handle:
        return TrainConfig(**tomllib.load(handle))


ROLE_LOSS_NAMES = {
    "board": ROLE_BOARD,
    "move": ROLE_MOVE,
    "evaluation": ROLE_EVAL,
    "control": ROLE_CONTROL,
    "refuted": ROLE_REFUTED,
    "final_move": ROLE_FINAL_MOVE,
    "final_line": ROLE_FINAL_PV,
    "final_evaluation": ROLE_FINAL_EVAL,
}


def trace_loss_parts(model, inputs):
    """Return equal-weight next-token CE over all non-padding trace tokens."""
    trace_ids = inputs["trace_ids"]
    roles = inputs["trace_roles"]
    targets = trace_ids.long()
    logits = model.trace_logits(inputs["ids"], trace_ids)
    ce = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape(trace_ids.shape)
    generated = roles != ROLE_PAD
    loss = (ce * generated).sum() / generated.sum().clamp_min(1)
    return loss, ce, logits, generated


def trace_loss(model, inputs):
    """Equal-weight next-token CE over generated trace tokens."""
    loss, _, logits, generated = trace_loss_parts(model, inputs)
    return loss, logits, generated


def role_loss_totals(ce, roles):
    """Return per-role CE sums and token counts for diagnostics."""
    return {
        name: ((ce * (roles == role)).sum(), int((roles == role).sum()))
        for name, role in ROLE_LOSS_NAMES.items()
    }


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
    train_data = TracePositions(data_dir, "train")
    val_data = TracePositions(data_dir, "val")
    eval_collate = TraceCollator()
    data_identity = trace_data_identity(data_dir)
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
    examples_seen = 0
    board_tokens = 0
    trace_tokens = 0
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
        board_tokens = checkpoint["board_tokens"]
        trace_tokens = checkpoint["trace_tokens"]
        examples_seen = checkpoint["examples_seen"]
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
        loss_sum, token_count, correct, count = 0.0, 0, 0, 0
        role_totals = {name: [0.0, 0] for name in ROLE_LOSS_NAMES}
        for start in range(0, len(indices), config.batch_size):
            selected = indices[start : start + config.batch_size]
            inputs = {
                key: value.to(device)
                for key, value in eval_collate([data[i] for i in selected]).items()
            }
            with autocast():
                _, ce, logits, generated = trace_loss_parts(model, inputs)
            loss_sum += float((ce * generated).sum())
            token_count += int(generated.sum())
            for name, (total, tokens) in role_loss_totals(
                ce, inputs["trace_roles"]
            ).items():
                role_totals[name][0] += float(total)
                role_totals[name][1] += tokens
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
        values = {
            f"{prefix}/loss": loss_sum / max(token_count, 1),
            f"{prefix}/final_move_top1": correct / count,
            f"{prefix}/rows": count,
        }
        values.update(
            {
                f"{prefix}/role/{name}_loss": total / tokens
                for name, (total, tokens) in role_totals.items()
                if tokens
            }
        )
        return values

    probe_indices = (
        torch.linspace(
            0,
            len(train_data) - 1,
            min(config.train_probe_positions, len(train_data)),
        )
        .long()
        .tolist()
        if config.train_probe_positions
        else []
    )

    def validate():
        values = evaluate(
            val_data, range(min(config.val_positions, len(val_data))), "val"
        )
        if config.train_probe_positions:
            values.update(evaluate(train_data, probe_indices, "train_probe"))
            values["generalization/loss_gap"] = (
                values["val/loss"] - values["train_probe/loss"]
            )
            values["generalization/final_move_top1_gap"] = (
                values["val/final_move_top1"] - values["train_probe/final_move_top1"]
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
            board_tokens=board_tokens,
            trace_tokens=trace_tokens,
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
            train_collate = TraceCollator()
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
            step_loss_sum = torch.zeros((), device=device)
            step_tokens = 0
            for micro, inputs in enumerate(loader):
                if micro >= steps_per_epoch * config.accumulation_steps:
                    break
                if epoch == start_epoch and micro < next_micro:
                    continue
                board_lengths = (inputs["ids"] != PAD).sum(1).numpy()
                trace_lengths = (inputs["trace_ids"] != PAD).sum(1).numpy()
                board_tokens += int(board_lengths.sum())
                trace_tokens += int(trace_lengths.sum())
                examples_seen += config.batch_size
                inputs = {
                    key: value.to(device, non_blocking=True)
                    for key, value in inputs.items()
                }
                with autocast():
                    _, ce, _, generated = trace_loss_parts(model, inputs)
                micro_sum = (ce * generated).sum()
                micro_sum.backward()
                step_loss_sum += micro_sum.detach()
                step_tokens += int(generated.sum())
                if (micro + 1) % config.accumulation_steps:
                    continue
                # The optimizer batch is one token-mean loss over all
                # supervised tokens, not the mean of per-microbatch means.
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(max(step_tokens, 1))
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
                    val_metrics = validate()
                if should_validate or step % config.log_every == 0:
                    values = {
                        "step": step,
                        "epoch": epoch,
                        "train/loss": (step_loss_sum / max(step_tokens, 1)).item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/lr_fraction": optimizer.param_groups[0]["lr"]
                        / config.learning_rate,
                        "train/grad_norm": grad_norm.item(),
                        "train/embedding_grad_norm": token_grad_norm,
                        "train/rows_per_s": (step - initial_step)
                        * config.batch_size
                        * config.accumulation_steps
                        / (time.monotonic() - start_time),
                        "train/board_tokens": board_tokens,
                        "train/trace_tokens": trace_tokens,
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
                step_loss_sum.zero_()
                step_tokens = 0
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
        "train/board_tokens": board_tokens,
        "train/trace_tokens": trace_tokens,
        "train/examples": examples_seen,
        "train/data_passes": examples_seen / len(train_data),
        **validate(),
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
