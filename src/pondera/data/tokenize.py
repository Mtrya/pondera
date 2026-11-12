"""Fit a trace dictionary and encode authoritative text directly to memory maps."""

import json
import multiprocessing
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyarrow.parquet as pq

from pondera.chess.codec import encode_board
from pondera.chess.merge import MergeDictionary
from pondera.chess.vocabulary import CHESS_TOKENIZER_VERSION, CHESS_VOCAB_SIZE
from pondera.data.dataset import (
    HF_SCHEMA,
    load_manifest,
    parquet_sources,
    write_manifest,
)
from pondera.data.packing import PACKED_FORMAT_VERSION, PRESENTATION_CODES, SPLIT_CODES
from pondera.data.text import (
    MERGE_BOUNDARIES,
    TraceEncoder,
    trace_atoms,
    validate_trace_dictionary,
)
from pondera.data.trace import TRACE_GRAMMAR_VERSION


def merge_and_encode(source, output, config, *, max_document_tokens=131_072):
    """Fit the configured total vocabulary size, then encode the complete dataset.

    Corpus sampling and optional holdout affect vocabulary fitting only.
    The output keeps all source rows and their training split membership.
    """
    from pondera.data.merge_profile import peak_rss_bytes, render_report

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    settings = asdict(config)
    options = {
        key: value
        for key, value in settings.items()
        if key not in {"vocab_size", "batch_size"}
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    document, dictionary = _fitted_dictionary(
        checkpoint_path(output), source, config.vocab_size, settings, options
    )
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as work:
        dataset = Path(work) / "dataset"
        summary = encode_dataset(
            source,
            dataset,
            dictionary,
            batch_size=config.batch_size,
            workers=config.workers,
            max_document_tokens=max_document_tokens,
        )
        profile = dataset / "profile"
        profile.mkdir()
        document["settings"] = settings
        document["timing"]["encoding_seconds"] = summary["seconds"]
        document["timing"]["workflow_seconds"] = time.monotonic() - started
        document["timing"]["workflow_peak_rss_bytes"] = peak_rss_bytes()
        (profile / "curve.json").write_text(json.dumps(document, indent=2) + "\n")
        (profile / "profile.md").write_text(render_report(document))
        dataset.rename(output)
    return {
        **summary,
        "output": str(output),
        "workflow_seconds": time.monotonic() - started,
    }


def checkpoint_path(output):
    """Where a completed vocabulary fit is recorded beside its output."""
    return Path(output).parent / f"{Path(output).name}.fit-checkpoint.json"


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
    source = Path(source).resolve()
    fitting["source"] = {
        "path": str(source),
        "manifest": load_manifest(source / "release.json") if source.is_dir() else None,
    }
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
    temporary = checkpoint.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record) + "\n")
    temporary.replace(checkpoint)
    return document, dictionary


def _encode_shard(encoder, task):
    """Write one shard's token stream with bounded Arrow and token buffers."""
    path, expected_rows, split, directory, batch_size, check_header = task
    file = pq.ParquetFile(path)
    for field in HF_SCHEMA:
        if file.schema_arrow.field(field.name) != field:
            raise ValueError(f"Shared field {field.name} has an invalid type")
    columns = ["fen", "repetition", "trace_text", "presentation"]
    if "split" in file.schema_arrow.names:
        columns.append("split")
    elif split not in SPLIT_CODES:
        raise ValueError(f"No dataset split for {path}")
    directory = Path(directory)
    directory.mkdir()
    offsets, splits, presentations = [0], [], []
    base_atoms = trace_tokens = 0
    with (directory / "tokens.u16").open("wb") as handle:
        for batch in file.iter_batches(
            columns=columns, batch_size=batch_size, use_threads=False
        ):
            for row in batch.to_pylist():
                label = row.get("split", split)
                if label not in SPLIT_CODES:
                    raise ValueError(f"Unknown dataset split in {path}: {label}")
                atoms = trace_atoms(
                    row["trace_text"],
                    expected_fen=row["fen"] if check_header else None,
                    expected_repetition=row["repetition"] if check_header else None,
                )
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
        "directory": str(directory),
        "rows": len(splits),
        "base_atoms": base_atoms,
        "trace_tokens": trace_tokens,
    }


_TOKENIZE_ENCODER = None


def _tokenize_pool_init(entries):
    global _TOKENIZE_ENCODER
    _TOKENIZE_ENCODER = TraceEncoder(MergeDictionary(entries))


def _tokenize_task(task):
    return _encode_shard(_TOKENIZE_ENCODER, task)


def encode_dataset(
    source,
    output,
    dictionary=MergeDictionary(),
    *,
    batch_size=128,
    workers=1,
    max_document_tokens=131_072,
):
    """Encode local or HF text directly into the final memory-mapped layout.

    Local atom arrays are diagnostic caches and are never read here. Workers
    stage shard streams on disk; concatenation follows source order without
    moving token arrays through process IPC. Publish only a complete output.
    """
    if batch_size < 1 or workers < 1 or max_document_tokens < 1:
        raise ValueError("Batch size, workers, and document cap must be positive")
    validate_trace_dictionary(dictionary)
    vocabulary = CHESS_VOCAB_SIZE + len(dictionary)
    if vocabulary > np.iinfo(np.uint16).max:
        raise ValueError("The uint16 layout needs vocabulary size below 65536")
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(source / "release.json") if source.is_dir() else {}
    if manifest:
        for field, expected in (
            ("grammar_version", TRACE_GRAMMAR_VERSION),
            ("tokenizer_version", CHESS_TOKENIZER_VERSION),
        ):
            if manifest.get(field) != expected:
                raise ValueError(f"Unsupported {field} in {source}")
    start = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as work:
        work = Path(work)
        dataset = work / "dataset"
        dataset.mkdir()
        tasks = [
            (
                path,
                expected,
                split,
                str(work / f"shard-{index:06d}"),
                batch_size,
                not manifest,
            )
            for index, (path, expected, split) in enumerate(parquet_sources(source))
        ]
        if workers > 1 and len(tasks) > 1:
            context = multiprocessing.get_context("spawn")
            with context.Pool(
                workers, initializer=_tokenize_pool_init, initargs=(dictionary.entries,)
            ) as pool:
                results = pool.map(_tokenize_task, tasks)
        else:
            encoder = TraceEncoder(dictionary)
            results = [_encode_shard(encoder, task) for task in tasks]
        position = rows = base_atoms = trace_tokens = 0
        with (
            (dataset / "tokens.u16").open("wb") as tokens,
            (dataset / "offsets.i64").open("wb") as offsets,
            (dataset / "split.u8").open("wb") as splits,
            (dataset / "presentation.u8").open("wb") as presentations,
        ):
            np.asarray([0], dtype=np.int64).tofile(offsets)
            for result in results:
                directory = Path(result["directory"])
                for name, destination in (
                    ("tokens.u16", tokens),
                    ("split.u8", splits),
                    ("presentation.u8", presentations),
                ):
                    with (directory / name).open("rb") as source_file:
                        shutil.copyfileobj(source_file, destination)
                boundaries = np.fromfile(directory / "offsets.i64", dtype=np.int64)
                (boundaries[1:] + position).tofile(offsets)
                position += int(boundaries[-1])
                rows += result["rows"]
                base_atoms += result["base_atoms"]
                trace_tokens += result["trace_tokens"]
                shutil.rmtree(directory)
        if not rows:
            raise ValueError("No documents to encode")
        write_manifest(
            dataset / "packing.json",
            {
                "format_version": PACKED_FORMAT_VERSION,
                "source": str(source),
                "source_dataset_id": manifest.get("dataset_id") or uuid4().hex,
                "grammar_version": TRACE_GRAMMAR_VERSION,
                "tokenizer_version": CHESS_TOKENIZER_VERSION,
                "merge_boundaries": MERGE_BOUNDARIES,
                "merge_entries": dictionary.entries,
                "vocabulary": vocabulary,
                "max_document_tokens": max_document_tokens,
                "documents": rows,
                "tokens": position,
            },
        )
        lengths = np.diff(np.fromfile(dataset / "offsets.i64", dtype=np.int64))
        split_codes = np.fromfile(dataset / "split.u8", dtype=np.uint8)
        presentation_codes = np.fromfile(dataset / "presentation.u8", dtype=np.uint8)
        summary = {
            "source": str(source),
            "documents": rows,
            "base_atoms": base_atoms,
            "trace_tokens": trace_tokens,
            "tokens": position,
            "ratio": trace_tokens / base_atoms if base_atoms else None,
            "entries": len(dictionary),
            "max_document_tokens": max_document_tokens,
            "ineligible_documents": int((lengths > max_document_tokens).sum()),
            "length": dict(
                zip(
                    ("p50", "p90", "p95", "p99", "max"),
                    map(float, np.quantile(lengths, (0.5, 0.9, 0.95, 0.99, 1))),
                )
            ),
            "splits": {
                name: int((split_codes == code).sum())
                for name, code in SPLIT_CODES.items()
            },
            "eligible_splits": {
                name: int(
                    ((split_codes == code) & (lengths <= max_document_tokens)).sum()
                )
                for name, code in SPLIT_CODES.items()
            },
            "presentations": {
                name: int((presentation_codes == code).sum())
                for name, code in PRESENTATION_CODES.items()
            },
            "seconds": time.monotonic() - start,
        }
        write_manifest(dataset / "stats.json", summary)
        dataset.rename(output)
    return {**summary, "output": str(output)}
