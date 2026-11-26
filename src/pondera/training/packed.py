"""Minimal trainer for the frozen packed batch contract.

Muon owns hidden matrices, AdamW the rest (training/muon.py). Each optimizer
update has a global budget of whole-document packs. Each pack's next-token
CE is weighted by its share of the update's labeled tokens. Resume starts
at the saved number of consumed documents in a fixed selection and shuffled
epoch order. Data-parallel ranks consume interleaved packs, and the DDP
gradient average is rescaled to the global labeled-token count.
"""

import importlib
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from pondera.chess.merge import MergeDictionary
from pondera.data.packing import (
    PackedDocuments,
    assemble_batch,
    pack_batches,
    packed_batch_indices,
    select_documents,
)
from pondera.models import PonderaModel
from pondera.models.internals import Internals, attach, detach
from pondera.training.config import DEFAULT_CONFIG_PATH, TrainConfig, load_config
from pondera.training.muon import build_optimizers


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


def _lr_scale(progress, warmup_fraction, min_lr_ratio):
    if progress < warmup_fraction:
        return progress / warmup_fraction
    decay = (progress - warmup_fraction) / (1 - warmup_fraction)
    return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * decay)) / 2


def _move(batch, device):
    moved = {
        "host_input_ids": batch["input_ids"],
        "input_ids": torch.as_tensor(
            batch["input_ids"], dtype=torch.long, device=device
        ),
        "labels": torch.as_tensor(batch["labels"], dtype=torch.long, device=device),
        "cu_seqlens": batch["cu_seqlens"].tolist(),
        "labeled_tokens": batch["tokens"] - batch["documents"],
    }
    return moved


def _forward(model, batch):
    raw_model = _module(model)
    device_type = raw_model.token_embedding.weight.device.type
    options = {}
    if raw_model.navigation and "host_input_ids" in batch:
        options["navigation_reads"] = raw_model.prepare_navigation(
            batch["host_input_ids"], batch["cu_seqlens"]
        )
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=device_type in {"cuda", "npu"},
    ):
        return model(batch["input_ids"], batch["cu_seqlens"], **options)


def _loss(model, batch):
    if not batch["input_ids"].numel():
        # An idle rank participates in DDP with zero gradient and no presented tokens.
        moved = {
            "input_ids": batch["input_ids"].new_zeros(1),
            "cu_seqlens": [0, 1],
        }
        return _forward(model, moved).float().sum() * 0.0, 0
    logits = _forward(model, batch)
    labeled = batch["labeled_tokens"]
    if not labeled:
        # A nonempty unsupervised batch still participates in the same forward/DDP.
        return logits.float().sum() * 0.0, 0
    logits = logits.float().view(-1, logits.shape[-1])
    labels = batch["labels"].view(-1)
    loss = F.cross_entropy(logits, labels, ignore_index=-100)
    return loss, labeled


def _validation_loss_sum(model, moved, boundaries):
    logits = _forward(model, moved).float()
    token_losses = F.cross_entropy(
        logits, moved["labels"], ignore_index=-100, reduction="none"
    )
    return np.add.reduceat(
        token_losses.cpu().numpy().astype(np.float64), boundaries[:-1]
    )


@torch.no_grad()
def validation_loss(
    model,
    documents,
    budget,
    device,
    indices,
    rank=0,
    world_size=1,
    *,
    collect_internals=False,
    document_losses=None,
):
    model = _module(model)
    was_training = model.training
    model.eval()
    batches = pack_batches(documents.lengths, indices, budget)
    internals = Internals() if collect_internals and rank == 0 and batches else None
    total_loss = 0.0
    total_labeled = 0
    report = {}
    if document_losses is not None:
        positions = {int(document): i for i, document in enumerate(indices)}
        sums = np.zeros(len(indices), dtype=np.float32)
    try:
        for i, batch in enumerate(batches[rank::world_size]):
            probe = i == 0 and internals is not None
            packed = assemble_batch(documents, list(batch))
            moved = _move(packed, device)
            if probe:
                attach(model, internals)
            try:
                if document_losses is None:
                    loss, labeled = _loss(model, moved)
                    total_loss += float(loss) * labeled
                    total_labeled += labeled
                else:
                    losses = _validation_loss_sum(model, moved, packed["cu_seqlens"])
                    sums[[positions[int(index)] for index in batch]] = losses
            finally:
                if probe:
                    detach(model)
            if probe:
                report = {
                    "documents": [int(index) for index in batch],
                    "tokens": packed["tokens"],
                    "metrics": internals.report(),
                }
        if document_losses is None:
            total_loss = _reduce(total_loss, device)
            total_labeled = _reduce(total_labeled, device)
        else:
            if world_size > 1:
                sums = _reduce(sums, device)
            total_loss = sum(float(value) for value in sums)
            total_labeled = int((documents.lengths[indices] - 1).sum())
            if rank == 0:
                document_losses.extend(
                    {
                        "document": int(index),
                        "tokens": int(documents.lengths[index] - 1),
                        "nll_sum": float(loss),
                    }
                    for index, loss in zip(indices, sums, strict=True)
                )
    finally:
        model.train(was_training)
    if not total_labeled:
        raise ValueError("No next-token labels in validation documents")
    return total_loss / total_labeled, total_labeled, report


def _write_validation(output_dir, record, rows, dataset_id):
    destination = output_dir / f"validation-{record['update']:08d}"
    destination.mkdir(exist_ok=True)
    (destination / "documents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    summary = {
        "update": record["update"],
        "presented_documents": record["presented_documents"],
        "documents": len(rows),
        "labeled_tokens": record["val_labeled_tokens"],
        "loss": record["val_loss"],
        "data_identity": dataset_id,
    }
    temporary = destination / "summary.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(destination / "summary.json")


def save_training_checkpoint(
    path,
    model,
    optimizers,
    run_config,
    micro_step,
    update,
    presented_documents,
    indices,
    keep_last,
    swanlab_run_id=None,
):
    if keep_last < 1:
        raise ValueError("checkpoint_keep_last must be positive")
    model = _module(model)
    payload = {
        "format_version": 3,
        **run_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dicts": [optimizer.state_dict() for optimizer in optimizers],
        "micro_step": micro_step,
        "update": update,
        "presented_documents": presented_documents,
        "swanlab_run_id": swanlab_run_id,
        "document_indices": indices.tolist(),
    }
    snapshot = path.with_name(f"checkpoint-{update:08d}.pth")
    temporary = snapshot.with_suffix(snapshot.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, snapshot)
    latest = path.with_suffix(path.suffix + ".link")
    latest.unlink(missing_ok=True)
    latest.symlink_to(snapshot.name)
    os.replace(latest, path)
    snapshots = sorted(
        (int(candidate.stem.removeprefix("checkpoint-")), candidate)
        for candidate in path.parent.glob("checkpoint-*.pth")
        if candidate.stem.removeprefix("checkpoint-").isdigit()
    )
    eligible = [candidate for step, candidate in snapshots if step <= update]
    retained = set(eligible[-keep_last:])
    for _, candidate in snapshots:
        if candidate not in retained:
            candidate.unlink()


def _open_history(path, update):
    """Discard records after the checkpoint, including an interrupted final write."""
    handle = path.open("a+", buffering=1)
    handle.seek(0)
    while True:
        position = handle.tell()
        line = handle.readline()
        if not line:
            break
        if not line.endswith("\n") or json.loads(line)["update"] > update:
            handle.seek(position)
            handle.truncate()
            break
    return handle


def train(
    model,
    model_config,
    packed_dir,
    config: TrainConfig,
    output_dir,
    resume=None,
    device="cpu",
    track=True,
    *,
    stop_after_update=None,
    validation_internals=False,
    validation_document_steps=(),
):
    config.validate()
    if stop_after_update is not None and (
        type(stop_after_update) is not int or stop_after_update < 1
    ):
        raise ValueError("stop_after_update must be a positive integer")
    device, rank, world_size = _distributed(device)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and resume is None:
        raise FileExistsError(f"Use a new run directory or resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    documents = PackedDocuments(packed_dir)
    if model.navigation:
        model.bind_tokenizer(MergeDictionary(documents.manifest["merge_entries"]))
    cap = (
        documents.manifest["max_document_tokens"]
        if config.max_document_tokens is None
        else config.max_document_tokens
    )
    indices = select_documents(
        documents, seed=config.seed, count=config.documents, max_document_tokens=cap
    )
    validation_indices = np.flatnonzero(documents.split == 1)
    validation_budget = max(
        config.max_pack_tokens,
        int(documents.lengths[validation_indices].max())
        if len(validation_indices)
        else 0,
    )
    if not len(indices):
        raise ValueError("No eligible training documents selected")
    maximum = int(documents.lengths[indices].max())
    if maximum > config.max_pack_tokens:
        raise ValueError(
            f"Selected document length {maximum} exceeds max_pack_tokens={config.max_pack_tokens}"
        )
    run_config = {
        "arch": model.architecture,
        "model_config": model_config,
        "train_config": asdict(config),
        "selection": {
            "documents": len(indices),
            "max_document_tokens": cap,
            "validation_documents": len(validation_indices),
        },
        "data_identity": {
            key: documents.manifest[key]
            for key in ("source_dataset_id", "documents", "tokens")
        },
        "tokenizer_identity": {
            key: documents.manifest[key]
            for key in (
                "grammar_version",
                "tokenizer_version",
                "merge_boundaries",
                "merge_entries",
            )
        },
    }
    if run_config["data_identity"]["source_dataset_id"] is None:
        raise ValueError("Packed training data needs a dataset_id from encoding")
    model.to(device)
    model.activation_checkpointing = config.activation_checkpointing
    optimizers = build_optimizers(
        model,
        muon_lr=config.muon_lr,
        adamw_lr=config.adamw_lr,
        adamw_weight_decay=config.adamw_weight_decay,
        optimizer=config.optimizer,
    )
    micro_step = update = presented_documents = 0
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=True)
        if state["format_version"] != 3:
            raise ValueError("Unsupported training checkpoint format")
        for key in ("arch", "model_config", "data_identity", "tokenizer_identity"):
            if state[key] != run_config[key]:
                raise ValueError(f"Resume {key} differs from checkpoint")
        if state["document_indices"] != indices.tolist():
            raise ValueError("Resume document selection differs from checkpoint")
        saved_config = state["train_config"]
        for key in ("seed", "max_pack_tokens", "packs_per_update"):
            if saved_config.get(key) != run_config["train_config"][key]:
                raise ValueError(f"Resume {key} differs from checkpoint")
        saved_init_seed = saved_config.get("init_seed")
        if saved_init_seed is None:
            saved_init_seed = saved_config["seed"]
        if saved_init_seed != config.initialization_seed:
            raise ValueError("Resume init_seed differs from checkpoint")
        if saved_config["optimizer"] != config.optimizer:
            raise ValueError("Resume optimizer differs from checkpoint")
        model.load_state_dict(state["model_state_dict"])
        for optimizer, saved in zip(
            optimizers, state["optimizer_state_dicts"], strict=True
        ):
            optimizer.load_state_dict(saved)
        for group in optimizers[1].param_groups:
            group["weight_decay"] = config.adamw_weight_decay
        micro_step, update = state["micro_step"], state["update"]
        presented_documents = state["presented_documents"]
    if stop_after_update is not None and update >= stop_after_update:
        raise ValueError("Checkpoint already reached the requested optimizer step")
    update_limit = min(
        (limit for limit in (config.updates, stop_after_update) if limit is not None),
        default=None,
    )
    swanlab_run_id = state.get("swanlab_run_id") if resume is not None else None
    schedule_documents = (
        int(len(indices) * config.presentations) if config.updates is None else None
    )
    target_documents = schedule_documents
    if config.max_presented_documents is not None:
        target_documents = (
            config.max_presented_documents
            if target_documents is None
            else min(target_documents, config.max_presented_documents)
        )
        if presented_documents > config.max_presented_documents:
            raise ValueError("Checkpoint already exceeds the document presentation cap")
    if world_size > 1:
        distributed_args = (
            {} if device.type == "cpu" else {"device_ids": [device.index]}
        )
        model = DistributedDataParallel(model, **distributed_args)
    model.train()
    tracker = None
    if rank == 0 and track:
        import swanlab

        logdir = output_dir / "swanlog"
        logdir.mkdir(exist_ok=True)
        tracker = swanlab.init(
            project="pondera-model-arch",
            experiment_name=output_dir.name,
            config=run_config,
            logdir=logdir,
            id=swanlab_run_id,
            resume="allow",
        )
        swanlab_run_id = tracker.public.run_id
    if rank == 0:
        (output_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    history = []
    history_log = (
        _open_history(output_dir / "history.jsonl", update) if rank == 0 else None
    )
    batch_stream = packed_batch_indices(
        documents,
        indices,
        config.seed,
        config.max_pack_tokens,
        start_document=presented_documents,
        presentations=config.presentations if schedule_documents is not None else None,
        max_documents=target_documents,
    )
    started = time.monotonic()
    try:
        while (update_limit is None or update < update_limit) and (
            target_documents is None or presented_documents < target_documents
        ):
            update_started = time.monotonic()
            batches = []
            total_labeled = 0
            total_tokens = 0
            last_epoch = 0
            packs_remaining = config.packs_per_update
            while packs_remaining and (
                target_documents is None or presented_documents < target_documents
            ):
                local_indices = []
                for batch_rank in range(min(world_size, packs_remaining)):
                    if (
                        target_documents is not None
                        and presented_documents >= target_documents
                    ):
                        break
                    document_indices, last_epoch = next(batch_stream)
                    tokens = int(documents.lengths[document_indices].sum())
                    total_tokens += tokens
                    presented_documents += len(document_indices)
                    micro_step += 1
                    packs_remaining -= 1
                    if batch_rank == rank:
                        local_indices = document_indices
                batch = assemble_batch(documents, local_indices)
                total_labeled += batch["tokens"] - batch["documents"]
                batches.append(batch)
            total_labeled = _reduce(total_labeled, device)
            if rank == 0:
                progress_path = output_dir / "progress.json"
                temporary = progress_path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(
                        {
                            "presented_documents": presented_documents,
                            "update": update + 1,
                        }
                    )
                    + "\n"
                )
                os.replace(temporary, progress_path)
            if not total_labeled:
                raise ValueError("No next-token labels in the training update")
            for optimizer in optimizers:
                optimizer.zero_grad()
            update_loss = torch.zeros((), device=device)
            for index, batch in enumerate(batches):
                moved = _move(batch, device)
                weight = moved["labeled_tokens"]
                context = (
                    model.no_sync()
                    if world_size > 1 and index + 1 < len(batches)
                    else nullcontext()
                )
                with context:
                    loss, _ = _loss(model, moved)
                    (loss * (weight * world_size / total_labeled)).backward()
                update_loss += loss.detach() * weight
            # Use the budget fraction completed by this update, including a short final batch.
            progress = (
                (update + 1) / config.updates
                if config.updates is not None
                else presented_documents / schedule_documents
            )
            lr_scale = _lr_scale(progress, config.warmup_fraction, config.min_lr_ratio)
            for optimizer, peak_lr in zip(
                optimizers, (config.muon_lr, config.adamw_lr)
            ):
                for group in optimizer.param_groups:
                    group["lr"] = peak_lr * lr_scale
                optimizer.step()
            update_loss = float(_reduce(update_loss, device)) / total_labeled
            if device.type in {"cuda", "npu"}:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                else:
                    torch.npu.synchronize(device)
            update_seconds = time.monotonic() - update_started
            update += 1
            record = {
                "update": update,
                "micro_step": micro_step,
                "loss": update_loss,
                "tokens": total_tokens,
                "labeled_tokens": total_labeled,
                "presentations": presented_documents / len(indices),
                "presented_documents": presented_documents,
                "epoch": last_epoch,
                "update_seconds": update_seconds,
                "elapsed_seconds": time.monotonic() - started,
                "tokens_per_second": total_tokens / update_seconds,
                "muon_lr": optimizers[0].param_groups[0]["lr"],
                "adamw_lr": optimizers[1].param_groups[0]["lr"],
            }
            if update in validation_document_steps or (
                config.val_every
                and (
                    update % config.val_every == 0
                    or presented_documents == target_documents
                    or update == stop_after_update
                )
            ):
                validation_started = time.monotonic()
                document_losses = [] if update in validation_document_steps else None
                (
                    record["val_loss"],
                    record["val_labeled_tokens"],
                    internals,
                ) = validation_loss(
                    model,
                    documents,
                    validation_budget,
                    device,
                    validation_indices,
                    rank,
                    world_size,
                    collect_internals=validation_internals,
                    document_losses=document_losses,
                )
                if internals:
                    record["internals"] = internals
                if rank == 0 and document_losses is not None:
                    _write_validation(
                        output_dir,
                        record,
                        document_losses,
                        run_config["data_identity"]["source_dataset_id"],
                    )
                record["validation_seconds"] = time.monotonic() - validation_started
            history.append(record)
            if tracker is not None:
                tracker.log(
                    {key: value for key, value in record.items() if key != "internals"},
                    step=update,
                )
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
                    run_config,
                    micro_step,
                    update,
                    presented_documents,
                    indices,
                    keep_last=config.checkpoint_keep_last,
                    swanlab_run_id=swanlab_run_id,
                )
        if rank == 0:
            save_training_checkpoint(
                output_dir / "checkpoint.pth",
                model,
                optimizers,
                run_config,
                micro_step,
                update,
                presented_documents,
                indices,
                keep_last=config.checkpoint_keep_last,
                swanlab_run_id=swanlab_run_id,
            )
    finally:
        if history_log is not None:
            history_log.close()
        if tracker is not None:
            tracker.finish()
    if world_size > 1:
        dist.barrier()
    return {
        "updates": update,
        "micro_steps": micro_step,
        "presented_documents": presented_documents,
        "presentations": presented_documents / len(indices),
        "seconds": time.monotonic() - started,
        "history": history,
        "rank": rank,
        "world_size": world_size,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, type=Path)
    parser.add_argument("--packed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-presented-documents", type=int)
    parser.add_argument("--stop-after-update", type=int)
    parser.add_argument("--validation-internals", action="store_true")
    parser.add_argument("--validation-document-steps", nargs="+", type=int, default=())
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--track", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    model_settings, config = load_config(args.config)
    if args.max_presented_documents is not None:
        config.max_presented_documents = args.max_presented_documents
    model_config = {
        **model_settings,
        "vocab_size": PackedDocuments(args.packed).manifest["vocabulary"],
    }
    torch.manual_seed(config.initialization_seed)
    model = PonderaModel(**model_config)
    result = train(
        model,
        model_config,
        args.packed,
        config,
        args.output,
        resume=args.resume,
        device=args.device,
        track=args.track,
        stop_after_update=args.stop_after_update,
        validation_internals=args.validation_internals,
        validation_document_steps=args.validation_document_steps,
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
