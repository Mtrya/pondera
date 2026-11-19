"""Full-document next-token loss with per-document results for paired comparisons."""

import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pondera.data.packing import PackedDocuments, assemble_batch, pack_batches
from pondera.inference.trace import load_checkpoint


@torch.inference_mode()
def evaluate_loss(
    checkpoint, packed, *, device="cpu", batch_tokens=131072, loop_repetitions=None
):
    if device.startswith("npu"):
        importlib.import_module("torch_npu")
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
    for batch_indices in pack_batches(documents.lengths, indices, budget):
        batch = assemble_batch(documents, batch_indices)
        inputs = torch.as_tensor(batch["input_ids"], dtype=torch.long, device=device)
        labels = torch.as_tensor(batch["labels"], dtype=torch.long, device=device)
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type in {"cuda", "npu"},
        ):
            logits = predictor.model(inputs, batch["cu_seqlens"].tolist())
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
    tokens = sum(row["tokens"] for row in results)
    return {
        "loss": sum(row["nll_sum"] for row in results) / tokens,
        "documents": len(results),
        "labeled_tokens": tokens,
        "seconds": time.monotonic() - started,
        "device": device,
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
    result = evaluate_loss(
        args.checkpoint,
        args.packed,
        device=args.device,
        batch_tokens=args.batch_tokens,
        loop_repetitions=args.loop_repetitions,
    )
    args.output.mkdir(parents=True)
    rows = result.pop("document_losses")
    (args.output / "documents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
