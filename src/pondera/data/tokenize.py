"""Create or replace a trace dataset's training IDs from authoritative text."""

import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from pondera.chess.merge import MergeDictionary, MergeEncoder
from pondera.chess.vocabulary import CHESS_TOKENIZER_VERSION
from pondera.data.dataset import (
    DATA_FORMAT_VERSION,
    HF_SCHEMA,
    TRACE_SCHEMA,
    load_manifest,
    parquet_sources,
    write_manifest,
)
from pondera.data.text import trace_atoms
from pondera.data.trace import TRACE_GRAMMAR_VERSION


def merge_and_encode(source, output, config):
    """Fit the configured composite budget, then encode the complete dataset.

    Corpus sampling and optional holdout affect vocabulary fitting only.
    The output keeps all source rows and their training split membership.
    """
    from pondera.data.merge_profile import build_profile, peak_rss_bytes, render_report

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    settings = asdict(config)
    options = {
        key: value
        for key, value in settings.items()
        if key not in {"budget", "batch_size"}
    }
    document, dictionaries = build_profile([source], [config.budget], **options)
    dictionary = dictionaries[config.budget]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as work:
        dataset = Path(work) / "dataset"
        summary = tokenize_dataset(
            source, dataset, dictionary, batch_size=config.batch_size
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


def _replace(table, name, values, type):
    column = pa.array(values, type=type)
    index = table.schema.get_field_index(name)
    if index < 0:
        return table.append_column(name, column)
    return table.set_column(index, name, column)


def tokenize_dataset(source, output, dictionary=MergeDictionary(), *, batch_size=128):
    """Stream local releases or HF Parquet into a separate training artifact.

    Existing ID/length columns are replaced. All other columns, including
    local-only teacher annotations when present, pass through unchanged.
    """
    if batch_size < 1:
        raise ValueError("Tokenization batch size must be positive")
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(source / "release.json") if source.is_dir() else {}
    if manifest:
        if manifest.get("format_version") not in {9, DATA_FORMAT_VERSION}:
            raise ValueError(f"Unsupported dataset format in {source}")
        for field, expected in (
            ("grammar_version", TRACE_GRAMMAR_VERSION),
            ("tokenizer_version", CHESS_TOKENIZER_VERSION),
        ):
            if manifest.get(field) != expected:
                raise ValueError(f"Unsupported {field} in {source}")
    encoder = MergeEncoder(dictionary)
    start = time.monotonic()
    rows = base_atoms = tokens = 0
    shards = []
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as work:
        work = Path(work)
        for index, (path, expected_rows, split) in enumerate(parquet_sources(source)):
            file = pq.ParquetFile(path)
            unknown = set(file.schema_arrow.names) - set(TRACE_SCHEMA.names)
            if unknown:
                raise ValueError(f"Unknown dataset columns: {sorted(unknown)}")
            for field in HF_SCHEMA:
                if file.schema_arrow.field(field.name) != field:
                    raise ValueError(f"Shared field {field.name} has an invalid type")
            if "split" not in file.schema_arrow.names and split not in {"train", "val"}:
                raise ValueError(f"No dataset split for {path}")
            name = f"shard-{index:06d}"
            (work / name).mkdir()
            shard_rows = 0
            schema = file.schema_arrow
            for field in (
                TRACE_SCHEMA.field("trace_token_ids"),
                TRACE_SCHEMA.field("trace_len"),
            ):
                column = schema.get_field_index(field.name)
                schema = (
                    schema.append(field) if column < 0 else schema.set(column, field)
                )
            with pq.ParquetWriter(
                work / name / "data.parquet",
                schema,
                compression="zstd",
                compression_level=3,
            ) as writer:
                for batch in file.iter_batches(
                    batch_size=batch_size, use_threads=False
                ):
                    if "split" in batch.schema.names:
                        labels = set(batch.column("split").to_pylist())
                        if labels - {"train", "val"}:
                            raise ValueError(
                                f"Unknown dataset split in {path}: {labels}"
                            )
                    encoded = []
                    for row in batch.select(
                        ["fen", "repetition", "trace_text"]
                    ).to_pylist():
                        atoms = trace_atoms(
                            row["trace_text"],
                            expected_fen=row["fen"] if not manifest else None,
                            expected_repetition=row["repetition"]
                            if not manifest
                            else None,
                        )
                        encoded.append(encoder.encode(atoms).tolist())
                        base_atoms += len(atoms)
                    lengths = [len(ids) for ids in encoded]
                    table = _replace(
                        pa.Table.from_batches([batch]),
                        "trace_token_ids",
                        encoded,
                        pa.list_(pa.int32()),
                    )
                    table = _replace(table, "trace_len", lengths, pa.int32())
                    writer.write_table(table, row_group_size=64)
                    shard_rows += len(batch)
                    tokens += sum(lengths)
            if shard_rows != expected_rows:
                raise ValueError(f"Shard row count differs from its release: {path}")
            shard = {"path": name, "rows": shard_rows}
            if split is not None:
                shard["split"] = split
            shards.append(shard)
            rows += shard_rows
        write_manifest(
            work / "release.json",
            {
                **manifest,
                "format_version": DATA_FORMAT_VERSION,
                "grammar_version": TRACE_GRAMMAR_VERSION,
                "tokenizer_version": CHESS_TOKENIZER_VERSION,
                "merge_entries": dictionary.entries,
                "dataset_id": manifest.get("dataset_id") or uuid4().hex,
                "source": str(source),
                "shards": shards,
            },
        )
        summary = {
            "source": str(source),
            "rows": rows,
            "base_atoms": base_atoms,
            "tokens": tokens,
            "ratio": tokens / base_atoms if base_atoms else None,
            "entries": len(dictionary),
            "seconds": time.monotonic() - start,
        }
        write_manifest(work / "tokenization.json", summary)
        work.rename(output)
    return summary
