"""Policy-only SL probe on pre-tokenized Stockfish positions.

Reads the memmap files produced by data/probe/prepare.py, trains the v1 ~100M
encoder with a single cross-entropy loss on the Stockfish best move, and logs
loss / lr / grad norm / val top-1 / throughput to log/probe.log and swanlab
(project "pondera"). The swanlab run id is stored in checkpoints so a resumed
run appends to the same remote curves.

Resume: `--resume <ckpt>` restores model/optimizer/scheduler and skips already
trained micro batches. Data order is reproducible because each epoch uses its
own DataLoader with a dedicated generator seeded by (SEED + epoch).
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import swanlab
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from chess_core import UCI_MOVE_TO_IDX
from chess_core.tokenize import TOKENIZER_VERSION
from model import PonderaModel

REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "data" / "probe"
DEFAULT_CKPT_DIR = REPO / "ckpts" / "probe"
LOG_PATH = Path(os.environ.get("PROBE_LOG", REPO / "log" / "probe.log"))

SEED = 640
MODEL_CONFIG = {
    "num_blocks": 20,
    "hidden_size": 640,
    "intermediate_size": 1728,
    "num_heads": 8,
    "dropout": 0.05,
    "possible_moves": len(UCI_MOVE_TO_IDX),
    "dtype": torch.float32,
}

# The 20-block model keeps ~57MB of activations per sample (~15 fp32-sized
# tensors per block), so on a 12GB card batch 192 and up OOM even in forward;
# 128 is the largest safe micro batch. Accumulation makes up the effective batch.
MICRO_BATCH_SIZE = 128
ACCUMULATION_STEPS = 8
EPOCHS = 2
LEARNING_RATE = 3e-4
WARMUP_RATIO = 0.03
GRAD_CLIP = 1.0
NUM_WORKERS = 4
VAL_EVERY_STEPS = 500
VAL_POSITIONS = 2048
LOG_EVERY_STEPS = 50
SAVE_EVERY_STEPS = 2_000


class TokenizedPositions(Dataset):
    """Memmap-backed (token ids, move index) pairs."""

    def __init__(self, split: str):
        self.tokens = np.load(DATA_DIR / f"tokens_{split}.npy", mmap_mode="r")
        self.moves = np.load(DATA_DIR / f"moves_{split}.npy", mmap_mode="r")
        assert len(self.tokens) == len(self.moves), "tokens/moves length mismatch"

    def __len__(self) -> int:
        return len(self.moves)

    def __getitem__(self, idx: int):
        return self.tokens[idx].copy(), int(self.moves[idx])


def build_logger() -> logging.Logger:
    logger = logging.getLogger("probe")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_PATH)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


def save_checkpoint(
    step, epoch, model, optimizer, scheduler, num_params, logger, ckpt_dir, mark=None, swanlab_id=None
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = f"probe_step{step:06d}.pth" if mark is None else f"probe_{mark}.pth"
    path = ckpt_dir / name
    tmp = path.with_suffix(".tmp")
    payload = {
        "step": step,
        "epoch": epoch,
        "model_config": MODEL_CONFIG,
        "config": MODEL_CONFIG,  # evaluate.py reads this key
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "num_params": num_params,
    }
    if swanlab_id is not None:
        payload["swanlab_id"] = swanlab_id
    torch.save(payload, tmp)
    os.replace(tmp, path)
    logger.info(f"step={step:6d} checkpoint saved to {path}")
    print(f"checkpoint saved to {path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    parser.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR))
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY_STEPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)

    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TokenizedPositions("train")
    val_ds = TokenizedPositions("val")

    val_ids = torch.from_numpy(np.array(val_ds.tokens[:VAL_POSITIONS]))
    val_moves = torch.from_numpy(np.array(val_ds.moves[:VAL_POSITIONS]).astype(np.int64))

    model = PonderaModel(**MODEL_CONFIG).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    micro_per_epoch = len(train_ds) // MICRO_BATCH_SIZE
    steps_per_epoch = micro_per_epoch // ACCUMULATION_STEPS
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scheduler.base_lrs = [LEARNING_RATE for _ in optimizer.param_groups]

    logger = build_logger()
    logger.info(
        f"start | params={num_params / 1e6:.1f}M | positions_train={len(train_ds)} "
        f"positions_val={len(val_ds)} | micro_batch={MICRO_BATCH_SIZE} accum={ACCUMULATION_STEPS} "
        f"effective_batch={MICRO_BATCH_SIZE * ACCUMULATION_STEPS} | steps_per_epoch={steps_per_epoch} "
        f"total_steps={total_steps} warmup_steps={warmup_steps} lr={LEARNING_RATE}"
    )

    start_step = 0
    start_epoch = 0
    swanlab_id = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        assert ckpt["model_config"] == MODEL_CONFIG, "resume ckpt config mismatch"
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt["step"]
        start_epoch = ckpt["epoch"]
        swanlab_id = ckpt.get("swanlab_id")
        logger.info(f"resumed from {args.resume} at step={start_step} epoch={start_epoch}")
        print(f"resumed from {args.resume} at step={start_step} epoch={start_epoch}", flush=True)

    swanlab_run = swanlab.init(
        project="pondera",
        experiment_name="probe-15M-2ep",
        config={
            "model": {k: str(v) if isinstance(v, torch.dtype) else v for k, v in MODEL_CONFIG.items()},
            "num_params": num_params,
            "tokenizer_version": TOKENIZER_VERSION,
            "micro_batch": MICRO_BATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "epochs": EPOCHS,
            "lr": LEARNING_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "grad_clip": GRAD_CLIP,
            "seed": SEED,
            "positions_train": len(train_ds),
            "positions_val": len(val_ds),
            "total_steps": total_steps,
        },
        id=swanlab_id,
        resume="allow",
    )

    def evaluate() -> tuple[float, float]:
        model.eval()
        correct, loss_sum = 0, 0.0
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            for i in range(0, len(val_moves), MICRO_BATCH_SIZE):
                ids = val_ids[i : i + MICRO_BATCH_SIZE].to(device, non_blocking=True)
                targets = val_moves[i : i + MICRO_BATCH_SIZE].to(device, non_blocking=True)
                logits, _ = model(ids=ids)
                loss_sum += criterion(logits.float(), targets).item() * len(targets)
                correct += (logits.argmax(dim=-1) == targets).sum().item()
        model.train()
        return correct / len(val_moves), loss_sum / len(val_moves)

    model.train()
    step = start_step
    train_seconds = 0.0
    loss_window = 0.0
    micro_window = 0
    val_top1 = float("nan")
    val_loss = float("nan")
    optimizer.zero_grad()
    t_start = time.time()
    stop = False

    for epoch in range(start_epoch, EPOCHS):
        logger.info(f"epoch {epoch + 1}/{EPOCHS} start at step {step}")
        # Per-epoch loader with dedicated generator: reproducible data order
        # regardless of RNG history, so resume can skip within an epoch.
        train_loader = DataLoader(
            train_ds,
            batch_size=MICRO_BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4,
            generator=torch.Generator().manual_seed(SEED + epoch),
        )
        skip_micros = (step - epoch * steps_per_epoch) * ACCUMULATION_STEPS
        for micro_idx, (ids, targets) in enumerate(train_loader):
            if micro_idx < skip_micros:
                continue
            ids = ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            t_micro = time.time()
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits, _ = model(ids=ids)
                loss = criterion(logits.float(), targets)
            (loss / ACCUMULATION_STEPS).backward()

            loss_window += loss.item()
            micro_window += 1
            if (micro_idx + 1 - skip_micros) % ACCUMULATION_STEPS != 0:
                train_seconds += time.time() - t_micro
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            train_seconds += time.time() - t_micro
            step += 1

            if step % VAL_EVERY_STEPS == 0 or step == 1:
                val_top1, val_loss = evaluate()

            if step % LOG_EVERY_STEPS == 0 or step == 1:
                now = time.time()
                train_loss = loss_window / micro_window
                pos_per_s = (step - start_step) * MICRO_BATCH_SIZE * ACCUMULATION_STEPS / train_seconds
                message = (
                    f"step={step:6d} loss={train_loss:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} top1={val_top1:.4f} "
                    f"val_loss={val_loss:.4f} pos/s={pos_per_s:.1f} "
                    f"wall={now - t_start:.0f}s"
                )
                logger.info(message)
                print(message, flush=True)
                swanlab.log(
                    {
                        "train/loss": train_loss,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/grad_norm": grad_norm.item(),
                        "train/pos_per_s": pos_per_s,
                        "sys/gpu_mem_mb": torch.cuda.max_memory_allocated() / 1e6,
                        "val/top1": val_top1,
                        "val/loss": val_loss,
                        "epoch": epoch,
                    },
                    step=step,
                )
                loss_window = 0.0
                micro_window = 0

            if step % args.save_every == 0:
                save_checkpoint(
                    step, epoch, model, optimizer, scheduler, num_params, logger, ckpt_dir,
                    swanlab_id=swanlab_run.id,
                )

            if step >= total_steps:
                stop = True
                break
        if stop:
            break

    save_checkpoint(
        step, EPOCHS - 1, model, optimizer, scheduler, num_params, logger, ckpt_dir,
        mark="final", swanlab_id=swanlab_run.id,
    )
    logger.info(f"done | steps={step} elapsed={time.time() - t_start:.0f}s")
    print(f"done | steps={step} elapsed={time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
