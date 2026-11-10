"""Memory-mapped documents, deterministic selection, and whole-document batches.

Encoding writes tokens.u16, offsets.i64, split.u8, and presentation.u8 directly.
The token stream contains each root board prefix followed by its encoded trace.
Selection filters stored lengths by the recipe's cap, then takes a seeded
permutation prefix. Sampling shuffles that fixed selection once per epoch.
Progress counts documents, so a new presentation target cannot skip documents.

Batches contain int32 input_ids, within-document shifted labels (last position
-100), and int32 cu_seqlens. Whole documents are packed greedily without padding;
a selected document exceeding the microbatch budget is an error.
"""

import random
from pathlib import Path

import numpy as np

from pondera.data.dataset import load_manifest

PACKED_FORMAT_VERSION = 3
IGNORE_INDEX = -100
SPLIT_CODES = {"train": 0, "val": 1}
PRESENTATION_CODES = {"zero": 0, "full": 1, "random": 2}


def epoch_order(count, seed, epoch):
    """The document shuffle of one epoch of one selection."""
    order = list(range(count))
    random.Random(f"{seed}:packing:epoch:{epoch}").shuffle(order)
    return order


def pack_batches(lengths, documents_in_order, budget):
    """Greedily pack every document in order; never skip or truncate one."""
    if budget < 1:
        raise ValueError("token budget must be positive")
    batches = []
    batch, fill = [], 0
    for document in documents_in_order:
        length = int(lengths[document])
        if length > budget:
            raise ValueError(
                f"Document {document} has {length} tokens, exceeding microbatch budget {budget}"
            )
        if fill and fill + length > budget:
            batches.append(batch)
            batch, fill = [], 0
        batch.append(document)
        fill += length
    if batch:
        batches.append(batch)
    return batches


class PackedDocuments:
    """Memory-mapped view of an encoded directory, including ineligible rows."""

    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = load_manifest(directory / "packing.json")
        if self.manifest.get("format_version") != PACKED_FORMAT_VERSION:
            raise ValueError(f"Unsupported packed format in {directory}")
        self.directory = directory
        self.tokens = np.memmap(directory / "tokens.u16", np.uint16, mode="r")
        self.offsets = np.memmap(directory / "offsets.i64", np.int64, mode="r")
        self.split = np.memmap(directory / "split.u8", np.uint8, mode="r")
        self.presentation = np.memmap(directory / "presentation.u8", np.uint8, mode="r")
        self.lengths = np.diff(self.offsets)

    def __len__(self):
        return len(self.offsets) - 1

    def document(self, index):
        return np.asarray(
            self.tokens[self.offsets[index] : self.offsets[index + 1]],
            dtype=np.int32,
        )


def select_documents(
    documents, *, seed, count=None, max_document_tokens=None, split="train"
):
    """Select an eligible, nested permutation prefix from stored lengths."""
    cap = (
        documents.manifest["max_document_tokens"]
        if max_document_tokens is None
        else max_document_tokens
    )
    if cap < 1 or (count is not None and count < 1):
        raise ValueError("Document cap and selection count must be positive")
    indices = np.flatnonzero(
        (documents.split == SPLIT_CODES[split]) & (documents.lengths <= cap)
    )
    order = list(range(len(indices)))
    random.Random(f"{seed}:packing:subsets").shuffle(order)
    return indices[order[:count]]


def assemble_batch(documents, batch, epoch=None):
    """One contract batch: packed tokens, shifted labels, boundaries."""
    pieces = [documents.document(document) for document in batch]
    input_ids = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int32)
    cu_seqlens = np.zeros(len(batch) + 1, dtype=np.int32)
    np.cumsum([len(piece) for piece in pieces], out=cu_seqlens[1:])
    labels = np.empty(len(input_ids), dtype=np.int32)
    labels[:-1] = input_ids[1:]
    labels[cu_seqlens[1:] - 1] = IGNORE_INDEX
    result = {
        "input_ids": input_ids,
        "labels": labels,
        "cu_seqlens": cu_seqlens,
        "documents": len(batch),
        "tokens": len(input_ids),
    }
    if epoch is not None:
        result["epoch"] = epoch
    return result


def packed_batch_indices(
    documents,
    indices,
    seed,
    budget,
    start_document=0,
    presentations=None,
):
    """Yield (indices, epoch) from consumed-document progress without token IO.

    A fractional presentation takes a prefix of the epoch's shuffled order,
    rounded down to a whole document. Extending that target resumes at the
    next document even when the old target ended with a partial batch.
    """
    if start_document < 0:
        raise ValueError("start_document must be nonnegative")
    if not len(indices):
        raise ValueError("No eligible documents selected")
    if presentations is not None and presentations <= 0:
        raise ValueError("presentations must be positive")
    maximum = int(documents.lengths[indices].max())
    if maximum > budget:
        raise ValueError(
            f"Selected document length {maximum} exceeds microbatch budget {budget}"
        )
    target = None if presentations is None else int(len(indices) * presentations)
    consumed = start_document
    while target is None or consumed < target:
        epoch, position = divmod(consumed, len(indices))
        end = (
            len(indices)
            if target is None
            else min(len(indices), position + target - consumed)
        )
        order = epoch_order(len(indices), seed, epoch)[position:end]
        for batch in pack_batches(documents.lengths, indices[order], budget):
            consumed += len(batch)
            yield batch, epoch


def packed_batches(
    documents, indices, seed, budget, start_document=0, presentations=None
):
    """Materialize batches from the deterministic document sequence."""
    for batch, epoch in packed_batch_indices(
        documents, indices, seed, budget, start_document, presentations
    ):
        yield assemble_batch(documents, batch, epoch=epoch)
