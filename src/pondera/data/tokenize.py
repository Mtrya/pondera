"""Fit a trace dictionary and encode stored base atoms directly to memory maps."""

import fcntl
import json
import multiprocessing
import shutil
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from uuid import uuid4

import numpy as np

from pondera.chess.codec import encode_board
from pondera.chess.merge import MergeDictionary
from pondera.chess.vocabulary import CHESS_TOKENIZER_VERSION, CHESS_VOCAB_SIZE
from pondera.data.dataset import (
    iter_atom_batches,
    load_manifest,
    open_trace_shard,
    parquet_sources,
    read_release,
    write_manifest,
)
from pondera.data.packing import PACKED_FORMAT_VERSION, PRESENTATION_CODES, SPLIT_CODES
from pondera.data.trace import TRACE_GRAMMAR_VERSION
from pondera.data.trace_encoding import (
    MERGE_BOUNDARIES,
    TraceEncoder,
    validate_trace_dictionary,
)


def merge_and_encode(source, output, config, *, max_document_tokens=131_072):
    """Fit the configured total vocabulary size, then encode the complete dataset.

    Corpus sampling and optional holdout affect vocabulary fitting only.
    The output keeps all source rows and their training split membership.
    """
    from pondera.data.merge_profile import peak_rss_bytes, render_report

    output = Path(output)
    started = time.monotonic()
    settings = asdict(config)
    options = {
        key: value
        for key, value in settings.items()
        if key not in {"vocab_size", "batch_size"}
    }

    def write_profile(dataset, summary):
        profile = dataset / "profile"
        profile.mkdir()
        document["settings"] = settings
        document["timing"]["encoding_seconds"] = summary["seconds"]
        document["timing"]["workflow_seconds"] = time.monotonic() - started
        document["timing"]["workflow_peak_rss_bytes"] = peak_rss_bytes()
        (profile / "curve.json").write_text(json.dumps(document, indent=2) + "\n")
        (profile / "profile.md").write_text(render_report(document))

    with _encoding_lock(output):
        document, dictionary = _fitted_dictionary(
            checkpoint_path(output), source, config.vocab_size, settings, options
        )
        summary = _encode_dataset(
            source,
            output,
            dictionary,
            config.batch_size,
            config.workers,
            max_document_tokens,
            write_profile,
        )
    return {**summary, "workflow_seconds": time.monotonic() - started}


def checkpoint_path(output):
    """Where a completed vocabulary fit is recorded beside its output."""
    return Path(output).parent / f"{Path(output).name}.fit-checkpoint.json"


def _source_key(source, manifest):
    if dataset_id := manifest.get("dataset_id"):
        return {"dataset_id": dataset_id}
    return {"path": str(Path(source).resolve())}


def _fitted_dictionary(checkpoint, source, vocab_size, settings, options):
    """Load a previous fit from its checkpoint, or run the fit and record it.

    Fitting costs hours at production scale while encoding can fail on
    transient storage or memory errors, so the fitted dictionary and its
    profile document persist next to the output before encoding starts. A
    checkpoint authorizes skipping the fit only when its source and every
    fitting-relevant setting match; worker counts and the encoding batch size
    do not change fit results.
    """
    fitting = {
        key: value
        for key, value in settings.items()
        if key not in {"batch_size", "workers"}
    }
    fitting["merge_boundaries"] = MERGE_BOUNDARIES
    fitting["input"] = "stored_atoms_in_release_order"
    source = Path(source).resolve()
    fitting["source"] = _source_key(source, read_release(source))
    if checkpoint.exists():
        record = json.loads(checkpoint.read_text())
        if record.get("settings") != fitting:
            raise ValueError(
                f"Fit checkpoint settings differ from the config: {checkpoint}"
            )
        dictionary = MergeDictionary(record["entries"])
        validate_trace_dictionary(dictionary)
        return record["document"], dictionary
    from pondera.data.merge_profile import build_profile

    document, dictionaries = build_profile([source], [vocab_size], **options)
    dictionary = dictionaries[vocab_size]
    record = {
        "settings": fitting,
        "document": document,
        "entries": [list(entry) for entry in dictionary.entries],
    }
    write_manifest(checkpoint, record)
    return document, dictionary


def _encode_shard(encoder, path, expected_rows, directory, batch_size):
    """Write one shard's token stream with bounded Arrow and token buffers."""
    columns = ["fen", "repetition", "presentation", "split"]
    file = open_trace_shard(path, expected_rows, columns)
    directory = Path(directory)
    directory.mkdir()
    offsets, splits, presentations = [0], [], []
    base_atoms = trace_tokens = 0
    with (directory / "tokens.u16").open("wb") as handle:
        for rows, atom_ids in iter_atom_batches(file, columns, batch_size):
            for index, row in enumerate(rows):
                label = row["split"]
                if label not in SPLIT_CODES:
                    raise ValueError(f"Unknown dataset split in {path}: {label}")
                atoms = atom_ids[index]
                trace = encoder.encode(atoms)
                board = encode_board(row["fen"], row["repetition"])
                board.astype(np.uint16).tofile(handle)
                trace.astype(np.uint16).tofile(handle)
                offsets.append(offsets[-1] + len(board) + len(trace))
                splits.append(SPLIT_CODES[label])
                presentations.append(PRESENTATION_CODES[row["presentation"]])
                base_atoms += len(atoms)
                trace_tokens += len(trace)
    if len(splits) != expected_rows:
        raise ValueError(f"Shard row count differs from its release: {path}")
    np.asarray(offsets, dtype=np.int64).tofile(directory / "offsets.i64")
    np.asarray(splits, dtype=np.uint8).tofile(directory / "split.u8")
    np.asarray(presentations, dtype=np.uint8).tofile(directory / "presentation.u8")
    return {
        "rows": len(splits),
        "tokens": offsets[-1],
        "base_atoms": base_atoms,
        "trace_tokens": trace_tokens,
    }


_TOKENIZE_ENCODER = None


def _tokenize_pool_init(entries):
    global _TOKENIZE_ENCODER
    _TOKENIZE_ENCODER = TraceEncoder(MergeDictionary(entries))


def _prepare_shard(encoder, task):
    path, expected, directory, batch_size = task
    if directory.is_dir():
        return True
    stage = directory.with_name(f".{directory.name}.{uuid4().hex}.tmp")
    try:
        result = _encode_shard(encoder, path, expected, stage, batch_size)
        write_manifest(stage / "manifest.json", result)
        stage.rename(directory)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return False


def _tokenize_task(task):
    return _prepare_shard(_TOKENIZE_ENCODER, task)


@contextmanager
def _encoding_lock(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(f".{output.name}.encode.lock")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if output.exists():
            raise FileExistsError(output)
        yield


def encode_dataset(
    source,
    output,
    dictionary=MergeDictionary(),
    *,
    batch_size=128,
    workers=1,
    max_document_tokens=131_072,
):
    """Encode curated atoms and assemble the completed shards in release order."""
    output = Path(output)
    with _encoding_lock(output):
        return _encode_dataset(
            source,
            output,
            dictionary,
            batch_size,
            workers,
            max_document_tokens,
        )


def _encode_dataset(
    source, output, dictionary, batch_size, workers, cap, write_profile=None
):
    if batch_size < 1 or workers < 1 or cap < 1:
        raise ValueError("Batch size, workers, and document cap must be positive")
    validate_trace_dictionary(dictionary)
    vocabulary = CHESS_VOCAB_SIZE + len(dictionary)
    if vocabulary > np.iinfo(np.uint16).max:
        raise ValueError("The uint16 layout needs vocabulary size below 65536")
    source = Path(source)
    manifest = read_release(source)
    if manifest:
        for field, expected in (
            ("grammar_version", TRACE_GRAMMAR_VERSION),
            ("tokenizer_version", CHESS_TOKENIZER_VERSION),
        ):
            if manifest.get(field) != expected:
                raise ValueError(f"Unsupported {field} in {source}")
    started = time.monotonic()
    workspace = output.with_name(f".{output.name}.encoded-shards")
    workspace.mkdir(exist_ok=True)
    identity = {
        "format_version": PACKED_FORMAT_VERSION,
        "source": _source_key(source, manifest),
        "grammar_version": TRACE_GRAMMAR_VERSION,
        "tokenizer_version": CHESS_TOKENIZER_VERSION,
        "merge_boundaries": MERGE_BOUNDARIES,
        "merge_entries": dictionary.entries,
    }
    marker = workspace / "manifest.json"
    if marker.is_file():
        if load_manifest(marker) != json.loads(json.dumps(identity)):
            raise ValueError(
                f"Encoded shard workspace belongs to a different source or dictionary: {workspace}"
            )
    else:
        write_manifest(marker, identity)
    dataset = workspace / "assembly"
    if dataset.exists():
        shutil.rmtree(dataset)
    for abandoned in workspace.glob(".shard-*.tmp"):
        shutil.rmtree(abandoned, ignore_errors=True)

    def tasks():
        for index, (path, expected, _) in enumerate(parquet_sources(source, manifest)):
            yield path.resolve(), expected, workspace / f"shard-{index:06d}", batch_size

    reused = _prepare_shards(tasks(), workers, dictionary)
    shards_seconds = time.monotonic() - started
    dataset.mkdir()
    packed_metadata = {
        **identity,
        "source": str(source),
        "source_dataset_id": manifest.get("dataset_id") or uuid4().hex,
        "vocabulary": vocabulary,
        "max_document_tokens": cap,
    }
    assembly_started = time.monotonic()
    summary = _assemble(dataset, tasks(), packed_metadata)
    summary.update(
        reused_shards=reused,
        shard_seconds=shards_seconds,
        assembly_seconds=time.monotonic() - assembly_started,
        seconds=time.monotonic() - started,
    )
    write_manifest(dataset / "stats.json", summary)
    if write_profile is not None:
        write_profile(dataset, summary)
    dataset.rename(output)
    shutil.rmtree(workspace)
    return {**summary, "output": str(output)}


def _prepare_shards(tasks, workers, dictionary):
    if workers == 1:
        encoder = TraceEncoder(dictionary)
        return sum(_prepare_shard(encoder, task) for task in tasks)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_tokenize_pool_init,
        initargs=(dictionary.entries,),
    ) as pool:
        pending = set()
        reused = 0
        while True:
            pending.update(
                pool.submit(_tokenize_task, task)
                for task in islice(tasks, 2 * workers - len(pending))
            )
            if not pending:
                return reused
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            reused += sum(future.result() for future in done)


def _length_summary(histogram, rows):
    lengths = np.asarray(sorted(histogram), dtype=np.int64)
    cumulative = np.cumsum([histogram[int(length)] for length in lengths])
    ranks = (rows - 1) * np.asarray([0.5, 0.9, 0.95, 0.99, 1])
    lower = lengths[np.searchsorted(cumulative, np.floor(ranks), side="right")]
    upper = lengths[np.searchsorted(cumulative, np.ceil(ranks), side="right")]
    return dict(
        zip(("p50", "p90", "p95", "p99", "max"), lower + (upper - lower) * (ranks % 1))
    )


def _assemble(dataset, tasks, metadata):
    """Stream complete shard streams and rebase their offsets for publication."""
    cap = metadata["max_document_tokens"]
    position = rows = base_atoms = trace_tokens = staged_bytes = 0
    splits, eligible_splits, presentations, length_counts = (
        Counter() for _ in range(4)
    )
    with (
        (dataset / "tokens.u16").open("wb") as tokens,
        (dataset / "offsets.i64").open("wb") as offsets,
        (dataset / "split.u8").open("wb") as split_file,
        (dataset / "presentation.u8").open("wb") as presentation_file,
    ):
        np.asarray([0], dtype=np.int64).tofile(offsets)
        for _, expected, directory, _ in tasks:
            result = load_manifest(directory / "manifest.json")
            boundaries = np.fromfile(directory / "offsets.i64", dtype=np.int64)
            lengths = np.diff(boundaries)
            with (directory / "tokens.u16").open("rb") as handle:
                shutil.copyfileobj(handle, tokens, length=1024 * 1024)
            split_codes = np.fromfile(directory / "split.u8", dtype=np.uint8)
            presentation_codes = np.fromfile(
                directory / "presentation.u8", dtype=np.uint8
            )
            split_codes.tofile(split_file)
            presentation_codes.tofile(presentation_file)
            (boundaries[1:] + position).tofile(offsets)
            position += result["tokens"]
            rows += result["rows"]
            base_atoms += result["base_atoms"]
            trace_tokens += result["trace_tokens"]
            staged_bytes += 2 * result["tokens"] + boundaries.nbytes + 2 * expected
            unique, counts = np.unique(lengths, return_counts=True)
            length_counts.update(dict(zip(unique.tolist(), counts.tolist())))
            for name, code in SPLIT_CODES.items():
                matching = split_codes == code
                splits[name] += int(matching.sum())
                eligible_splits[name] += int((matching & (lengths <= cap)).sum())
            for name, code in PRESENTATION_CODES.items():
                presentations[name] += int((presentation_codes == code).sum())
    if not rows:
        raise ValueError("No documents to encode")
    write_manifest(
        dataset / "packing.json", {**metadata, "documents": rows, "tokens": position}
    )
    return {
        "source": metadata["source"],
        "documents": rows,
        "base_atoms": base_atoms,
        "trace_tokens": trace_tokens,
        "tokens": position,
        "ratio": trace_tokens / base_atoms if base_atoms else None,
        "entries": len(metadata["merge_entries"]),
        "max_document_tokens": cap,
        "ineligible_documents": rows - sum(eligible_splits.values()),
        "length": _length_summary(length_counts, rows),
        "splits": dict(splits),
        "eligible_splits": dict(eligible_splits),
        "presentations": dict(presentations),
        "staged_array_bytes": staged_bytes,
        "packed_array_bytes": 2 * position + 8 * (rows + 1) + 2 * rows,
    }
