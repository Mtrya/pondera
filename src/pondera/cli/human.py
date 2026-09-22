"""Build the human position dataset."""

import argparse
import os


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deduplicated human-move chess dataset from nsarrazin/lichess-games-*."
    )
    parser.add_argument("--dataset-name", default="nsarrazin/lichess-games-2023-01")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default=".local/data/human")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--games-per-batch", type=int, default=2048)
    parser.add_argument("--bucket-count", type=int, default=2048)
    parser.add_argument("--max-games", type=int, default=20_000_000)
    parser.add_argument("--skip-games", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.001)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    from pondera.data.human import run

    run(args)
