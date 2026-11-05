"""Create or replace a trace dataset's training IDs from authoritative text."""

import argparse
import json
import tempfile
import time
from pathlib import Path

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
        dictionary.write(work / "merged_vocabulary.txt")
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
            writer = None
            try:
                for batch in file.iter_batches(
                    batch_size=batch_size, use_threads=False
                ):
                    encoded = []
                    for text in batch.column("trace_text").to_pylist():
                        atoms = trace_atoms(text)
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
                    if writer is None:
                        writer = pq.ParquetWriter(
                            work / name / "data.parquet",
                            table.schema,
                            compression="zstd",
                            compression_level=3,
                        )
                    writer.write_table(table, row_group_size=64)
                    shard_rows += len(batch)
                    tokens += sum(lengths)
            finally:
                if writer is not None:
                    writer.close()
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
                "merge_dictionary": "merged_vocabulary.txt",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vocabulary", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    dictionary = (
        MergeDictionary.read(args.vocabulary) if args.vocabulary else MergeDictionary()
    )
    print(
        json.dumps(
            tokenize_dataset(
                args.input, args.output, dictionary, batch_size=args.batch_size
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
