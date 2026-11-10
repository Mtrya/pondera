"""Minimal trainer for the frozen packed batch contract.

Muon owns hidden matrices, AdamW the rest (training/muon.py). One optimizer
update consumes whole microbatches until its token target is met; each
microbatch's equal-weight next-token CE is normalized by that microbatch's
labeled positions and weighted by its share of the update's actual labeled
tokens. The batch sequence is a pure function of (subset, seed, budget,
step), so resume replays it exactly from the saved micro step. With data
parallelism the token target is global, ranks consume interleaved batches,
and the DDP gradient average is rescaled to the global labeled-token count.
"""

import importlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from pondera.chess.codec import decode_board, legal_move_indices
from pondera.chess.vocabulary import FINAL, MOVE
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


def _module(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def _distributed(device):
    device = torch.device(device)
    if device.type == "npu":
        importlib.import_module("torch_npu")
    if dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank = 0
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            backend = {"cuda": "nccl", "npu": "hccl"}.get(device.type, "gloo")
            dist.init_process_group(backend)
            rank = dist.get_rank()
    if world_size > 1 and device.type in {"cuda", "npu"}:
        local_rank = device.index
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        device = torch.device(device.type, local_rank)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        else:
            torch.npu.set_device(device)
    return device, rank, world_size


def _reduce(value, device, op=dist.ReduceOp.SUM):
    if not dist.is_initialized():
        return value
    tensor = torch.as_tensor(value, device=device)
    dist.all_reduce(tensor, op=op)
    return tensor.tolist()


def _move(batch, device):
    dtypes = {"input_ids": torch.long, "labels": torch.long, "cu_seqlens": torch.int32}
    return {
        key: torch.as_tensor(value, dtype=dtypes[key], device=device)
        for key, value in batch.items()
        if key in dtypes
    }


def _forward(model, batch):
    device_type = _module(model).token_embedding.weight.device.type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=device_type in {"cuda", "npu"},
    ):
        return model(batch["input_ids"], batch["cu_seqlens"])


def _loss(model, batch):
    logits = _forward(model, batch)
    labeled = int((batch["labels"] != -100).sum())
    loss = F.cross_entropy(
        logits.float().view(-1, logits.shape[-1]),
        batch["labels"].view(-1),
        ignore_index=-100,
    )
    return loss, labeled


@torch.no_grad()
def validation_loss(model, documents, budget, device, rank=0, world_size=1):
    indices = np.flatnonzero(documents.split == 1)
    model.eval()
    batches, _ = pack_batches(documents.lengths, indices, budget)
    total_loss = 0.0
    total_labeled = 0
    for batch in batches[rank::world_size]:
        moved = _move(assemble_batch(documents, list(batch)), device)
        loss, labeled = _loss(model, moved)
        total_loss += float(loss) * labeled
        total_labeled += labeled
    total_loss = _reduce(total_loss, device)
    total_labeled = _reduce(total_labeled, device)
    model.train()
    return total_loss / max(total_labeled, 1)


@torch.no_grad()
def zero_move_accuracy(model, documents, budget, device, rank=0, world_size=1):
    """Legal-move argmax accuracy at each zero document's FINAL position."""
    import chess

    indices = np.flatnonzero((documents.split == 1) & (documents.presentation == 0))
    examples = []
    for index in indices[rank::world_size]:
        document = documents.document(index)
        final_positions = np.flatnonzero(document == FINAL)
        if len(final_positions) != 1 or final_positions[0] + 1 >= len(document):
            raise ValueError("A zero document needs one FINAL and a following decision")
        final = int(final_positions[0])
        fen, _ = decode_board(document[:final])
        target = int(document[final + 1])
        legal = [MOVE + move for move in legal_move_indices(chess.Board(fen))]
        if target not in legal:
            raise ValueError("The zero decision is not a standalone legal move token")
        examples.append((document[: final + 1], target, legal))

    model.eval()
    correct = 0
    batch = []
    batch_tokens = 0

    def flush():
        nonlocal batch, batch_tokens, correct
        if not batch:
            return
        input_ids = np.concatenate([example[0] for example in batch])
        cu_seqlens = np.zeros(len(batch) + 1, dtype=np.int32)
        np.cumsum([len(example[0]) for example in batch], out=cu_seqlens[1:])
        logits = _forward(
            model, _move({"input_ids": input_ids, "cu_seqlens": cu_seqlens}, device)
        )
        for index, (_, target, legal) in enumerate(batch):
            scores = logits[int(cu_seqlens[index + 1] - 1)].float()
            ids = torch.as_tensor(legal, device=device)
            predicted = legal[int(scores[ids].argmax())]
            correct += predicted == target
        batch = []
        batch_tokens = 0

    for example in examples:
        length = len(example[0])
        if batch and batch_tokens + length > budget:
            flush()
        batch.append(example)
        batch_tokens += length
    flush()
    correct, count = _reduce([correct, len(examples)], device)
    model.train()
    return {
        "correct": correct,
        "count": count,
        "accuracy": correct / count if count else 0.0,
    }


def save_training_checkpoint(
    path, model, optimizers, model_config, micro_step, update, config, manifest
):
    model = _module(model)
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
    device, rank, world_size = _distributed(device)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and resume is None:
        raise FileExistsError(f"Use a new run directory or resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
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
    model.to(device)
    if world_size > 1:
        distributed_args = (
            {} if device.type == "cpu" else {"device_ids": [device.index]}
        )
        model = DistributedDataParallel(model, **distributed_args)
    model.train()
    tracker = None
    if rank == 0 and config.track:
        import swanlab

        tracker = swanlab.init(
            project="pondera-model-arch",
            config={"subset": subset, "model": model_config, **asdict(config)},
            resume="allow",
        )
    if rank == 0:
        (output_dir / "config.json").write_text(
            json.dumps(
                {"model": model_config, "train": asdict(config), "subset": subset},
                indent=2,
            )
            + "\n"
        )
    history = []
    history_log = (
        (output_dir / "history.jsonl").open(
            "a" if resume is not None else "w", buffering=1
        )
        if rank == 0
        else None
    )
    batch_stream = packed_batches(
        documents,
        indices,
        config.seed,
        config.microbatch_tokens,
        start_step=micro_step + rank,
        rank=rank,
        world_size=world_size,
    )
    started = time.monotonic()
    while update < config.updates:
        batches = []
        total_labeled = 0
        total_tokens = 0
        last_epoch = 0
        while total_tokens < config.update_tokens:
            batch = next(batch_stream)
            labeled = int((batch["labels"] != -100).sum())
            step_tokens, step_labeled = _reduce([batch["tokens"], labeled], device)
            total_tokens += step_tokens
            total_labeled += step_labeled
            micro_step += world_size
            batches.append((batch, labeled))
            last_epoch = batch.get("epoch", 0)
        for optimizer in optimizers:
            optimizer.zero_grad()
        update_loss = 0.0
        for batch, labeled in batches:
            moved = _move(batch, device)
            loss, _ = _loss(model, moved)
            (loss * (labeled * world_size / total_labeled)).backward()
            update_loss += float(loss.detach()) * labeled
        update_loss = _reduce(update_loss, device) / total_labeled
        last_epoch = _reduce(last_epoch, device, op=dist.ReduceOp.MAX)
        for optimizer in optimizers:
            optimizer.step()
        update += 1
        record = {
            "update": update,
            "micro_step": micro_step,
            "loss": update_loss,
            "tokens": total_tokens,
            "labeled_tokens": total_labeled,
            "epoch": last_epoch,
        }
        if config.val_every and update % config.val_every == 0:
            record["val_loss"] = validation_loss(
                model, documents, config.microbatch_tokens, device, rank, world_size
            )
            zero_metrics = zero_move_accuracy(
                model, documents, config.microbatch_tokens, device, rank, world_size
            )
            record["val_move_accuracy"] = zero_metrics["accuracy"]
            record["val_zero_documents"] = zero_metrics["count"]
        history.append(record)
        if tracker is not None:
            tracker.log(record, step=update)
        if history_log is not None:
            history_log.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if (
            rank == 0
            and config.checkpoint_every
            and update % config.checkpoint_every == 0
        ):
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
    if rank == 0:
        history_log.close()
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
    if world_size > 1:
        dist.barrier()
    return {
        "updates": update,
        "micro_steps": micro_step,
        "seconds": time.monotonic() - started,
        "history": history,
        "rank": rank,
        "world_size": world_size,
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
    if result["rank"] == 0:
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "history"},
                indent=2,
            )
        )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
