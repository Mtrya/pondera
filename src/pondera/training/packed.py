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
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from pondera.data.packing import (
    PackedDocuments,
    assemble_batch,
    load_subsets,
    pack_batches,
    packed_batch_indices,
    presentation_batch_count,
)
from pondera.training.muon import build_optimizers


@dataclass
class TrainConfig:
    muon_lr: float = 0.02
    adamw_lr: float = 3e-4
    weight_decay: float = 0.0
    microbatch_tokens: int = 65_536
    update_tokens: int = 524_288
    presentations: float = 1.0
    updates: int | None = None
    val_every: int = 10
    checkpoint_every: int = 50
    seed: int = 640
    track: bool = True


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
    return {
        "input_ids": torch.as_tensor(
            batch["input_ids"], dtype=torch.long, device=device
        ),
        "labels": torch.as_tensor(batch["labels"], dtype=torch.long, device=device),
        "cu_seqlens": batch["cu_seqlens"].tolist(),
        "labeled_tokens": batch["tokens"] - batch["documents"],
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
    if not batch["input_ids"].numel():
        # An idle rank participates in DDP with zero gradient and no presented tokens.
        moved = {
            "input_ids": batch["input_ids"].new_zeros(1),
            "cu_seqlens": [0, 1],
        }
        return _forward(model, moved).float().sum() * 0.0, 0
    logits = _forward(model, batch)
    labeled = batch["labeled_tokens"]
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


def save_training_checkpoint(
    path,
    model,
    optimizers,
    run_config,
    micro_step,
    update,
    presented_documents,
    indices,
    swanlab_run_id=None,
):
    model = _module(model)
    payload = {
        "format_version": 2,
        **run_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dicts": [optimizer.state_dict() for optimizer in optimizers],
        "micro_step": micro_step,
        "update": update,
        "presented_documents": presented_documents,
        "swanlab_run_id": swanlab_run_id,
        "subset_indices": indices.tolist(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


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
    run_config = {
        "arch": model.architecture,
        "model_config": model_config,
        "train_config": asdict(config),
        "subset": subset,
        "data_identity": {
            key: documents.manifest[key]
            for key in ("source_dataset_id", "documents", "tokens")
        },
        "tokenizer_identity": {
            key: documents.manifest[key]
            for key in ("grammar_version", "tokenizer_version", "merge_entries")
        },
    }
    if run_config["data_identity"]["source_dataset_id"] is None:
        raise ValueError("Packed training data needs a dataset_id from encoding")
    model.to(device)
    optimizers = build_optimizers(
        model,
        muon_lr=config.muon_lr,
        adamw_lr=config.adamw_lr,
        weight_decay=config.weight_decay,
    )
    micro_step = update = presented_documents = 0
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=True)
        if state["format_version"] != 2:
            raise ValueError("Unsupported training checkpoint format")
        for key in ("arch", "model_config", "data_identity", "tokenizer_identity"):
            if state[key] != run_config[key]:
                raise ValueError(f"Resume {key} differs from checkpoint")
        if state["subset_indices"] != indices.tolist():
            raise ValueError("Resume subset differs from checkpoint")
        for key in ("seed", "microbatch_tokens"):
            if state["train_config"][key] != run_config["train_config"][key]:
                raise ValueError(f"Resume {key} differs from checkpoint")
        model.load_state_dict(state["model_state_dict"])
        for optimizer, saved in zip(optimizers, state["optimizer_state_dicts"]):
            learning_rates = [group["lr"] for group in optimizer.param_groups]
            optimizer.load_state_dict(saved)
            for group, learning_rate in zip(optimizer.param_groups, learning_rates):
                group["lr"] = learning_rate
                group["weight_decay"] = config.weight_decay
        micro_step, update = state["micro_step"], state["update"]
        presented_documents = state["presented_documents"]
    swanlab_run_id = state.get("swanlab_run_id") if resume is not None else None
    final_step = (
        presentation_batch_count(
            documents,
            indices,
            config.seed,
            config.microbatch_tokens,
            config.presentations,
        )
        if config.updates is None
        else None
    )
    if world_size > 1:
        distributed_args = (
            {} if device.type == "cpu" else {"device_ids": [device.index]}
        )
        model = DistributedDataParallel(model, **distributed_args)
    model.train()
    tracker = None
    if rank == 0 and config.track:
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
        config.microbatch_tokens,
        start_step=micro_step,
        presentations=config.presentations if final_step is not None else None,
    )
    started = time.monotonic()
    while (config.updates is not None and update < config.updates) or (
        final_step is not None and micro_step < final_step
    ):
        update_started = time.monotonic()
        batches = []
        total_labeled = 0
        total_tokens = 0
        last_epoch = 0
        while total_tokens < config.update_tokens and (
            final_step is None or micro_step < final_step
        ):
            local_indices = []
            for batch_rank in range(world_size):
                if final_step is not None and micro_step >= final_step:
                    break
                document_indices, last_epoch = next(batch_stream)
                tokens = int(documents.lengths[document_indices].sum())
                total_tokens += tokens
                total_labeled += tokens - len(document_indices)
                presented_documents += len(document_indices)
                micro_step += 1
                if batch_rank == rank:
                    local_indices = document_indices
            batch = assemble_batch(documents, local_indices)
            batches.append((batch, batch["tokens"] - batch["documents"]))
        for optimizer in optimizers:
            optimizer.zero_grad()
        update_loss = torch.zeros((), device=device)
        for index, (batch, labeled) in enumerate(batches):
            moved = _move(batch, device)
            context = (
                model.no_sync()
                if world_size > 1 and index + 1 < len(batches)
                else nullcontext()
            )
            with context:
                loss, _ = _loss(model, moved)
                (loss * (labeled * world_size / total_labeled)).backward()
            update_loss += loss.detach() * labeled
        for optimizer in optimizers:
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
            "epoch": last_epoch,
            "update_seconds": update_seconds,
            "tokens_per_second": total_tokens / update_seconds,
        }
        if config.val_every and (
            update % config.val_every == 0 or micro_step == final_step
        ):
            record["val_loss"] = validation_loss(
                model, documents, config.microbatch_tokens, device, rank, world_size
            )
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
                run_config,
                micro_step,
                update,
                presented_documents,
                indices,
                swanlab_run_id=swanlab_run_id,
            )
    if rank == 0:
        history_log.close()
        save_training_checkpoint(
            output_dir / "checkpoint.pth",
            model,
            optimizers,
            run_config,
            micro_step,
            update,
            presented_documents,
            indices,
            swanlab_run_id=swanlab_run_id,
        )
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
                f"--{field.replace('_', '-')}",
                action=argparse.BooleanOptionalAction,
                default=default,
            )
        else:
            parser.add_argument(
                f"--{field.replace('_', '-')}",
                type=int if field == "updates" else type(default),
                default=default,
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
