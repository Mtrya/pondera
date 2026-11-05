"""Export the shared eleven-column Hugging Face projection of a trace release."""

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pondera.data.dataset import (
    HF_COLUMNS,
    HF_SCHEMA,
    TRACE_SCHEMA,
    load_manifest,
)

SPLIT_NAMES = {"train": "train", "val": "validation"}


def export_hf(release, output, batch_size=1024):
    """Write deterministic train/validation Parquet files without machine arrays."""
    if batch_size < 1:
        raise ValueError("Export batch size must be positive")
    release = Path(release)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = load_manifest(release / "release.json")
    if not manifest or not manifest.get("shards"):
        raise ValueError(f"Release has no completed shards: {release}")
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "data").mkdir(parents=True)
    writers = {}
    counts = Counter()
    try:
        for shard in manifest["shards"]:
            path = release / shard["path"] / "data.parquet"
            source = pq.ParquetFile(path)
            if source.schema_arrow != TRACE_SCHEMA:
                raise ValueError(
                    f"Shard schema differs from the release contract: {path}"
                )
            if source.metadata.num_rows != shard["rows"]:
                raise ValueError(f"Shard row count differs from its release: {path}")
            for batch in source.iter_batches(
                batch_size=batch_size, columns=[*HF_COLUMNS, "split"], use_threads=False
            ):
                rows = batch.to_pylist()
                unknown = {row["split"] for row in rows} - set(SPLIT_NAMES)
                if unknown:
                    raise ValueError(f"Unknown release split: {sorted(unknown)[0]}")
                for local_split, hf_split in SPLIT_NAMES.items():
                    selected = [row for row in rows if row["split"] == local_split]
                    if not selected:
                        continue
                    if hf_split not in writers:
                        destination = temporary / "data" / f"{hf_split}-00000.parquet"
                        writers[hf_split] = pq.ParquetWriter(
                            destination,
                            HF_SCHEMA,
                            compression="zstd",
                            compression_level=3,
                            use_dictionary=["presentation", "turn", "final_move"],
                        )
                    writers[hf_split].write_table(
                        pa.Table.from_pylist(selected, schema=HF_SCHEMA)
                    )
                    counts[hf_split] += len(selected)
        if not counts:
            raise ValueError("Release contains no rows")
        for writer in writers.values():
            writer.close()
        files = [
            {
                "path": f"data/{split}-00000.parquet",
                "rows": counts[split],
                "bytes": (temporary / "data" / f"{split}-00000.parquet").stat().st_size,
            }
            for split in ("train", "validation")
            if split in counts
        ]
        os.replace(temporary, output)
    except BaseException:
        for writer in writers.values():
            writer.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"rows": dict(counts), "files": files, "columns": list(HF_COLUMNS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    print(json.dumps(export_hf(args.release, args.output, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
