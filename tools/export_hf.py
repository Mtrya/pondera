"""Export the shared eleven-column Hugging Face projection of a trace release."""

import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pondera.chess.vocabulary import CHESS_TOKENIZER_VERSION
from pondera.data.dataset import (
    DATA_FORMAT_VERSION,
    HF_COLUMNS,
    HF_SCHEMA,
    TRACE_SCHEMA,
    parquet_sources,
    read_release,
)
from pondera.data.trace import TRACE_GRAMMAR_VERSION

SPLIT_NAMES = {"train": "train", "val": "validation"}
MAX_PART_BYTES = 256 * 1024**2
ROW_GROUP_ROWS = 1024


class _SplitParts:
    def __init__(self, directory, split, max_bytes):
        self.directory = directory
        self.split = split
        self.max_bytes = max_bytes
        self.files = []
        self.buffer = []
        self.buffer_bytes = 0
        self.group = []
        self.group_rows = 0

    def add(self, batch):
        while batch.num_rows:
            take = min(ROW_GROUP_ROWS - self.group_rows, batch.num_rows)
            self.group.append(batch.slice(0, take))
            self.group_rows += take
            batch = batch.slice(take)
            if self.group_rows == ROW_GROUP_ROWS:
                self._buffer_group()

    def _buffer_group(self):
        group = pa.Table.from_batches(self.group, schema=HF_SCHEMA).combine_chunks()
        self.group = []
        self.group_rows = 0
        if self.buffer and self.buffer_bytes + group.nbytes > self.max_bytes:
            self._flush()
        self.buffer.append(group)
        self.buffer_bytes += group.nbytes
        if self.buffer_bytes >= self.max_bytes:
            self._flush()

    def _flush(self):
        self._write(pa.concat_tables(self.buffer))
        self.buffer.clear()
        self.buffer_bytes = 0

    def _write(self, table):
        name = f"{self.split}-{len(self.files):05d}.parquet"
        destination = self.directory / name
        pq.write_table(
            table,
            destination,
            row_group_size=ROW_GROUP_ROWS,
            compression="zstd",
            compression_level=3,
            use_dictionary=["presentation", "turn", "final_move"],
        )
        size = destination.stat().st_size
        if size > self.max_bytes and table.num_rows > 1:
            # Check the closed file, including its footer. An oversized document
            # occupies a part on its own because rows are indivisible.
            middle = table.num_rows // 2
            self._write(table.slice(0, middle))
            self._write(table.slice(middle))
            return
        self.files.append(
            {"path": f"data/{name}", "rows": table.num_rows, "bytes": size}
        )

    def close(self):
        if self.group:
            self._buffer_group()
        if self.buffer:
            self._flush()


def export_hf(release, output, batch_size=1024, *, max_part_bytes=MAX_PART_BYTES):
    """Write ordered split parts, bounded by bytes except for single rows."""
    if batch_size < 1 or max_part_bytes < 1:
        raise ValueError("Export batch size and part size must be positive")
    release, output = Path(release), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = read_release(release)
    if not manifest or not manifest.get("shards"):
        raise ValueError(f"Release has no completed shards: {release}")
    for field, expected in (
        ("format_version", DATA_FORMAT_VERSION),
        ("grammar_version", TRACE_GRAMMAR_VERSION),
        ("tokenizer_version", CHESS_TOKENIZER_VERSION),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"Unsupported release {field}: {manifest.get(field)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as work:
        stage = Path(work) / "export"
        directory = stage / "data"
        directory.mkdir(parents=True)
        splits = {
            label: _SplitParts(directory, name, max_part_bytes)
            for label, name in SPLIT_NAMES.items()
        }
        for path, expected, _ in parquet_sources(release, manifest):
            source = pq.ParquetFile(path)
            if source.schema_arrow != TRACE_SCHEMA:
                raise ValueError(
                    f"Shard schema differs from the release contract: {path}"
                )
            if source.metadata.num_rows != expected:
                raise ValueError(f"Shard row count differs from its release: {path}")
            for batch in source.iter_batches(
                batch_size=batch_size, columns=[*HF_COLUMNS, "split"], use_threads=False
            ):
                labels = batch["split"]
                projected = batch.select(HF_COLUMNS)
                selected_rows = 0
                for label, parts in splits.items():
                    selected = projected.filter(pc.equal(labels, label))
                    if selected.num_rows:
                        parts.add(selected)
                        counts[parts.split] += selected.num_rows
                        selected_rows += selected.num_rows
                if selected_rows != batch.num_rows:
                    raise ValueError(f"Unknown release split in {path}")
        if not counts:
            raise ValueError("Release contains no rows")
        for parts in splits.values():
            parts.close()
        stage.rename(output)
    files = [file for parts in splits.values() for file in parts.files]
    return {"rows": dict(counts), "files": files, "columns": list(HF_COLUMNS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-part-bytes", type=int, default=MAX_PART_BYTES)
    args = parser.parse_args()
    print(
        json.dumps(
            export_hf(
                args.release,
                args.output,
                args.batch_size,
                max_part_bytes=args.max_part_bytes,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
