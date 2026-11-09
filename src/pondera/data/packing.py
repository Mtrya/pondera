"""Packed training streams: document assembly, memmap layout, batch contract.

A tokenized dataset (``python -m pondera.data encode`` output) is packed once
into memory-mapped arrays; the online training path never touches text. A
document is the board segment — ``encode_board(fen, repetition)`` atoms —
immediately followed by the row's stored trace tokens. No separator token:
the board segment's fixed shape (piece atoms plus eight metadata atoms) makes
the boundary implicit.

Layout of a packed directory:

- ``tokens.u16``: the flat uint16 token stream, all documents concatenated in
  dataset (manifest shard) order. Requires vocabulary size <= 65535.
- ``offsets.i64``: int64 document boundaries, ``len(documents) + 1`` entries,
  starting at 0. Document ``i`` occupies ``[offsets[i], offsets[i+1])``.
- ``split.u8``: per document, 0 = train / 1 = val.
- ``presentation.u8``: per document, 0 = zero / 1 = full / 2 = random.
- ``packing.json``: format version, source dataset identity, vocabulary
  identity (merge entries hash), counts, content hashes.
- ``subsets.json``: nested data-scale manifests over train document indices
  (seeded permutation prefixes; the largest group is every train document).
- ``stats.json``: length distributions, drop counts, and reference packing
  statistics per data-scale group.

Batch contract (what the trainer consumes):

- ``input_ids``: int32 ``[T]``, the packed tokens of one microbatch. ``T`` is
  the actual token count: the token budget is an upper bound, never padded to.
- ``labels``: int32 ``[T]``; ``labels[i] == input_ids[i + 1]`` inside a
  document, and ``-100`` (ignore) at each document's last position. Labels
  never cross documents. Loss covers the whole document including the board
  segment, one equal-weight next-token CE per labeled position.
- ``cu_seqlens``: int32 ``[D + 1]`` cumulative document boundaries;
  ``cu_seqlens[0] == 0``, ``cu_seqlens[D] == T``. On the softmax side this is
  the block-diagonal causal boundary (no L×L mask is ever materialized); on
  the linear-attention/loop side it is the state-zeroing boundary.
- ``epoch``: int, the epoch this batch belongs to.

Packing rules: within one epoch the group's documents are shuffled once
(``random.Random(f"{seed}:packing:epoch:{epoch}")``) and packed greedily in
that order, whole documents only, never sorted by length; the epoch's final
batch may be partial. A document longer than the budget — or longer than
``MAX_DOCUMENT_TOKENS``, the standing guardrail — is skipped and counted, never
truncated. The global batch sequence is a pure function of (subset, seed,
budget): ``packed_batches`` plans each epoch once and can yield an interleaved
rank view, while ``packed_batch`` reconstructs any step directly, so resume
replays length metadata only, without reading skipped tokens.
"""

import hashlib
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pondera.chess.codec import encode_board
from pondera.chess.merge import MergeDictionary
from pondera.chess.vocabulary import (
    CHESS_TOKENIZER_VERSION,
    CHESS_VOCAB_SIZE,
    FINAL,
)
from pondera.data.dataset import (
    DATA_FORMAT_VERSION,
    load_manifest,
    parquet_sources,
    write_manifest,
)
from pondera.data.trace import TRACE_GRAMMAR_VERSION

PACKED_FORMAT_VERSION = 1
MAX_DOCUMENT_TOKENS = 128 * 1024
IGNORE_INDEX = -100
SPLIT_CODES = {"train": 0, "val": 1}
PRESENTATION_CODES = {"zero": 0, "full": 1, "random": 2}
# Data-scale groups; the largest is the full train split.
DATA_SCALE_GROUPS = {"50k": 50_000, "100k": 100_000, "200k": 200_000, "500k": None}
REFERENCE_BUDGETS = (131_072, 262_144, 524_288, 1_048_576)


def nested_subsets(count, sizes, seed):
    """Nested manifests over ``range(count)``: seeded permutation prefixes.

    A ``None`` size takes every index. Nested by construction: smaller groups
    are prefixes of the same permutation.
    """
    order = list(range(count))
    random.Random(f"{seed}:packing:subsets").shuffle(order)
    subsets = {}
    for name, size in sizes.items():
        take = count if size is None else min(size, count)
        subsets[name] = order[:take]
    return subsets


def epoch_order(count, seed, epoch):
    """The document shuffle of one epoch of one subset."""
    order = list(range(count))
    random.Random(f"{seed}:packing:epoch:{epoch}").shuffle(order)
    return order


def pack_batches(
    lengths, documents_in_order, budget, max_document_tokens=MAX_DOCUMENT_TOKENS
):
    """Greedy whole-document packing in the given order.

    Returns ``(batches, skipped)``: batches are lists of document indices and
    each holds at most ``budget`` tokens; skipped counts the documents left
    out for exceeding the budget or the drop guardrail. The final batch may
    be partial.
    """
    if budget < 1:
        raise ValueError("token budget must be positive")
    limit = min(budget, max_document_tokens)
    batches, skipped = [], 0
    batch, fill = [], 0
    for document in documents_in_order:
        length = int(lengths[document])
        if length > limit:
            skipped += 1
            continue
        if fill and fill + length > budget:
            batches.append(batch)
            batch, fill = [], 0
        batch.append(document)
        fill += length
    if batch:
        batches.append(batch)
    return batches, skipped


class PackedDocuments:
    """Memory-mapped view of a packed directory."""

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


def load_subsets(directory):
    """The packed directory's data-scale manifests as index arrays."""
    manifest = load_manifest(Path(directory) / "subsets.json")
    return {
        name: np.asarray(indices, dtype=np.int64)
        for name, indices in manifest["subsets"].items()
    }


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


def packed_batches(
    documents, indices, seed, budget, start_step=0, rank=0, world_size=1
):
    """Yield one rank's view of the global sequence, planning each epoch once."""
    if start_step < 0:
        raise ValueError("start_step must be nonnegative")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Require 0 <= rank < world_size")
    step = start_step
    epoch = 0
    while True:
        order = epoch_order(len(indices), seed, epoch)
        batches, _ = pack_batches(documents.lengths, indices[order], budget)
        if not batches:
            raise ValueError("No packable documents under this budget")
        if step >= len(batches):
            step -= len(batches)
            epoch += 1
            continue
        while step < len(batches):
            yield assemble_batch(documents, batches[step], epoch=epoch)
            step += world_size
        step -= len(batches)
        epoch += 1


def packed_batch(documents, indices, seed, budget, step):
    """The ``step``-th batch, replaying length metadata without reading skips."""
    return next(packed_batches(documents, indices, seed, budget, start_step=step))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantiles(lengths):
    if not len(lengths):
        return None
    values = np.quantile(lengths, (0.5, 0.9, 0.95, 0.99, 1.0))
    return dict(zip(("p50", "p90", "p95", "p99", "max"), (float(v) for v in values)))


def _group_statistics(lengths, indices, seed, budgets):
    order = indices[epoch_order(len(indices), seed, 0)]
    group = {
        "documents": len(indices),
        "tokens": int(lengths[indices].sum()) if len(indices) else 0,
        "length": _quantiles(lengths[indices]),
        "dropped_overlong": int((lengths[indices] > MAX_DOCUMENT_TOKENS).sum()),
        "reference_batches": {},
    }
    for budget in budgets:
        batches, skipped = pack_batches(lengths, order, budget)
        sizes = np.asarray([sum(lengths[b] for b in batch) for batch in batches])
        docs = np.asarray([len(batch) for batch in batches])
        group["reference_batches"][str(budget)] = {
            "batches": len(batches),
            "skipped_documents": skipped,
            "documents_per_batch": _quantiles(docs),
            "tokens_per_batch": _quantiles(sizes),
        }
    return group


def write_packed(
    source,
    output,
    *,
    seed,
    sizes=None,
    reference_budgets=REFERENCE_BUDGETS,
    max_document_tokens=MAX_DOCUMENT_TOKENS,
):
    """Pack a tokenized dataset directory into the memmap training layout.

    Reads the source's stored ``trace_token_ids`` (never text), assembles
    board + trace documents, and writes the flat streams, manifests, nested
    data-scale subsets, and statistics in one pass.
    """
    started = time.monotonic()
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(source / "release.json")
    for field, expected in (
        ("format_version", DATA_FORMAT_VERSION),
        ("grammar_version", TRACE_GRAMMAR_VERSION),
        ("tokenizer_version", CHESS_TOKENIZER_VERSION),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"Unsupported dataset {field} in {source}")
    dictionary = MergeDictionary(manifest.get("merge_entries", ()))
    for entry in dictionary.entries:
        if FINAL in entry:
            raise ValueError("A composite token contains FINAL")
    vocabulary = CHESS_VOCAB_SIZE + len(dictionary)
    if vocabulary > np.iinfo(np.uint16).max:
        raise ValueError("The uint16 layout needs vocabulary size below 65536")
    sizes = DATA_SCALE_GROUPS if sizes is None else sizes

    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as staging:
        staging = Path(staging)
        chunks = []
        boundaries = [0]
        splits = []
        presentations = []
        position = 0
        for path, expected_rows, shard_split in parquet_sources(source):
            file = pq.ParquetFile(path)
            columns = ["fen", "repetition", "presentation", "trace_token_ids"]
            columns.extend(
                name for name in ("split",) if name in file.schema_arrow.names
            )
            table = file.read(columns=columns, use_threads=False)
            if len(table) != expected_rows:
                raise ValueError(f"Shard row count differs from its release: {path}")
            documents = []
            for row in table.to_pylist():
                label = row["split"] if "split" in row else shard_split
                if label not in SPLIT_CODES:
                    raise ValueError(f"Unknown dataset split in {path}: {label}")
                trace = np.asarray(row["trace_token_ids"], dtype=np.uint16)
                if len(trace) and int(trace.max()) >= vocabulary:
                    raise ValueError(f"Token id exceeds the vocabulary in {path}")
                document = np.concatenate(
                    [
                        encode_board(row["fen"], row["repetition"]).astype(np.uint16),
                        trace,
                    ]
                )
                documents.append(document)
                position += len(document)
                boundaries.append(position)
                splits.append(SPLIT_CODES[label])
                presentations.append(PRESENTATION_CODES[row["presentation"]])
            if documents:
                chunks.append(np.concatenate(documents))
        tokens = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint16)
        offsets = np.asarray(boundaries, dtype=np.int64)
        splits = np.asarray(splits, dtype=np.uint8)
        presentations = np.asarray(presentations, dtype=np.uint8)

        tokens.tofile(staging / "tokens.u16")
        offsets.tofile(staging / "offsets.i64")
        splits.tofile(staging / "split.u8")
        presentations.tofile(staging / "presentation.u8")

        lengths = np.diff(offsets)
        train = np.flatnonzero(splits == SPLIT_CODES["train"])
        subsets = nested_subsets(len(train), sizes, seed)
        write_manifest(
            staging / "subsets.json",
            {
                "seed": seed,
                "subsets": {
                    name: train[indices].tolist() for name, indices in subsets.items()
                },
            },
        )
        entries_hash = hashlib.sha256(
            json.dumps(manifest.get("merge_entries", [])).encode()
        ).hexdigest()
        write_manifest(
            staging / "packing.json",
            {
                "format_version": PACKED_FORMAT_VERSION,
                "source": str(source),
                "source_dataset_id": manifest.get("dataset_id"),
                "grammar_version": TRACE_GRAMMAR_VERSION,
                "tokenizer_version": CHESS_TOKENIZER_VERSION,
                "merge_entries_sha256": entries_hash,
                "vocabulary": vocabulary,
                "seed": seed,
                "max_document_tokens": max_document_tokens,
                "documents": len(boundaries) - 1,
                "tokens": int(len(tokens)),
                "tokens_sha256": _sha256(staging / "tokens.u16"),
                "offsets_sha256": _sha256(staging / "offsets.i64"),
            },
        )
        packed = PackedDocuments(staging)
        statistics = {
            "documents": len(boundaries) - 1,
            "tokens": int(len(tokens)),
            "max_document_tokens": max_document_tokens,
            "dropped_overlong": int((lengths > max_document_tokens).sum()),
            "length": _quantiles(lengths),
            "splits": {
                name: int((splits == code).sum()) for name, code in SPLIT_CODES.items()
            },
            "presentations": {
                name: int((presentations == code).sum())
                for name, code in PRESENTATION_CODES.items()
            },
            "groups": {
                name: _group_statistics(
                    packed.lengths, np.asarray(train[indices]), seed, reference_budgets
                )
                for name, indices in subsets.items()
            },
        }
        write_manifest(staging / "stats.json", statistics)
        staging.rename(output)
    return {
        "output": str(output),
        "documents": statistics["documents"],
        "tokens": statistics["tokens"],
        "dropped_overlong": statistics["dropped_overlong"],
        "groups": {
            name: group["documents"] for name, group in statistics["groups"].items()
        },
        "seconds": time.monotonic() - started,
    }
