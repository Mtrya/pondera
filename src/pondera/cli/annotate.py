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
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--max-time", type=float, default=15.0)
    parser.add_argument(
        "--output", type=Path, default=Path(".local/data/annotations/labels.jsonl")
    )
    args = parser.parse_args()
    from pondera.data.annotation import AnnotationConfig, annotate_positions
    from pondera.data.parquet import sampled_batches

    def records():
        for batch in sampled_batches(
            args.files, ["fen", "next_move", "count"], args.rows
        ):
            for row in batch.to_pylist():
                yield {
                    "fen": row["fen"],
                    "played": row["next_move"],
                    "count": row["count"],
                }

    config = AnnotationConfig(
        engine=args.engine,
        depth=args.depth,
        multipv=args.multipv,
        hash_mb=args.hash_mb,
        max_time=args.max_time,
    )
    print(
        f"Annotated {annotate_positions(records(), args.output, config, args.workers)} positions"
    )
