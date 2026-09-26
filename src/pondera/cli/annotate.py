"""Annotate local position parquet files with Stockfish."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--engine", default="stockfish")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--multipv", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument(
        "--max-time",
        type=float,
        help="Optional per-position time cap; default is depth only",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(".local/data/annotations/labels.jsonl")
    )
    args = parser.parse_args()
    import pyarrow.parquet as pq

    from pondera.data.annotation import AnnotationConfig, annotate_positions
    from pondera.data.parquet import sampled_batches

    available = [set(pq.ParquetFile(path).schema_arrow.names) for path in args.files]
    columns = ["fen"] + [
        name
        for name in ("next_move", "count", "repetition")
        if all(name in schema for schema in available)
    ]
    if (
        any("repetition" in schema for schema in available)
        and "repetition" not in columns
    ):
        raise ValueError(
            "All inputs must agree on whether repetition counts are supplied"
        )

    def records():
        for batch in sampled_batches(args.files, columns, args.rows):
            for row in batch.to_pylist():
                if "next_move" in row:
                    row["played"] = row.pop("next_move")
                yield row

    config = AnnotationConfig(
        engine=args.engine,
        depth=args.depth,
        multipv=args.multipv,
        hash_mb=args.hash_mb,
        threads=args.threads,
        max_time=args.max_time,
    )
    print(
        f"Annotated {annotate_positions(records(), args.output, config, args.workers)} positions"
    )
