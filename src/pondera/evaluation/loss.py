"""Full-document next-token loss with per-document results for paired comparisons."""

import importlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from pondera.data.packing import PackedDocuments, assemble_batch, pack_batches
from pondera.inference.trace import load_checkpoint
from pondera.models.temporal_navigation import TokenValueNavigationModel


@torch.inference_mode()
def evaluate_loss(
    checkpoint, packed, *, device="cpu", batch_tokens=131072, loop_repetitions=None
):
    if device.startswith("npu"):
        importlib.import_module("torch_npu")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    target = torch.device(device)
    if world_size > 1 and target.type in {"cuda", "npu"}:
        index = target.index
        if index is None:
            index = int(os.environ.get("LOCAL_RANK", str(rank)))
        device = f"{target.type}:{index}"
        getattr(torch, target.type).set_device(device)
    started = time.monotonic()
    predictor = load_checkpoint(checkpoint, device, loop_repetitions=loop_repetitions)
    documents = PackedDocuments(packed)
    identity = predictor.metadata["tokenizer_identity"]
    for key in (
        "grammar_version",
        "tokenizer_version",
        "merge_boundaries",
        "merge_entries",
    ):
        if identity.get(key) != documents.manifest[key]:
            raise ValueError(f"Checkpoint and evaluation tokenizer differ in {key}")
    indices = np.flatnonzero(documents.split == 1)
    if not len(indices):
        raise ValueError("The packed dataset has no validation documents")
    budget = max(batch_tokens, int(documents.lengths[indices].max()))
    device_type = torch.device(device).type
    results = []
    batches = pack_batches(documents.lengths, indices, budget)
    for batch_indices in batches[rank::world_size]:
        batch = assemble_batch(documents, batch_indices)
        inputs = torch.as_tensor(batch["input_ids"], dtype=torch.long, device=device)
        labels = torch.as_tensor(batch["labels"], dtype=torch.long, device=device)
        offsets = batch["cu_seqlens"].tolist()
        options = {}
        if isinstance(predictor.model, TokenValueNavigationModel):
            options["navigation_reads"] = predictor.model.navigation.prepare(
                batch["input_ids"], offsets
            )
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type in {"cuda", "npu"},
        ):
            logits = predictor.model(inputs, offsets, **options)
        losses = (
            F.cross_entropy(logits.float(), labels, reduction="none", ignore_index=-100)
            .cpu()
            .numpy()
        )
        for index, start, stop in zip(
            batch_indices,
            batch["cu_seqlens"][:-1],
            batch["cu_seqlens"][1:],
            strict=True,
        ):
            count = int(stop - start - 1)
            results.append(
                {
                    "document": int(index),
                    "tokens": count,
                    "nll_sum": float(losses[start:stop].sum(dtype=np.float64)),
                }
            )
    if world_size > 1:
        gathered = [None] * world_size
        with torch.inference_mode(False):
            dist.all_gather_object(gathered, results)
        results = sorted(
            (row for rows in gathered for row in rows), key=lambda row: row["document"]
        )
    tokens = sum(row["tokens"] for row in results)
    return {
        "loss": sum(row["nll_sum"] for row in results) / tokens,
        "documents": len(results),
        "labeled_tokens": tokens,
        "seconds": time.monotonic() - started,
        "device": device,
        "world_size": world_size,
        "precision": "bfloat16" if device_type in {"cuda", "npu"} else "float32",
        "checkpoint": str(Path(checkpoint).resolve()),
        "data_identity": documents.manifest["source_dataset_id"],
        "model": predictor.metadata,
        "document_losses": results,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-tokens", type=int, default=131072)
    parser.add_argument("--loop-repetitions", type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        # Only small CPU result records are exchanged; model execution is independent.
        dist.init_process_group("gloo")
    rank = dist.get_rank() if distributed else 0
    try:
        result = evaluate_loss(
            args.checkpoint,
            args.packed,
            device=args.device,
            batch_tokens=args.batch_tokens,
            loop_repetitions=args.loop_repetitions,
        )
    finally:
        if distributed:
            dist.destroy_process_group()
    if rank:
        return
    args.output.mkdir(parents=True)
    rows = result.pop("document_losses")
    (args.output / "documents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
