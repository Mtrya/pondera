"""Build the stockfish position dataset."""

import argparse
import os


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Stockfish-labeled chess dataset from a deduplicated human-move source dataset."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-splits", default="all")
    parser.add_argument("--output-dir", default=".local/data/stockfish")
    parser.add_argument("--stockfish-path", default="stockfish")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--engine-threads", type=int, default=1)
    parser.add_argument("--rows-per-batch", type=int, default=32)
    parser.add_argument("--bucket-count", type=int, default=262144)
    parser.add_argument("--max-source-rows", type=int, default=None)
    parser.add_argument("--skip-source-rows", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.001)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    from pondera.data.stockfish import run

    run(args)
