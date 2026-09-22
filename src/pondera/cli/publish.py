"""Publish a prepared dataset to Hugging Face."""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload chess datasets (human and stockfish) to Hugging Face Hub."
    )
    parser.add_argument(
        "--human-path",
        default=".local/data/human/dataset",
        help="Path to human dataset directory produced by save_to_disk().",
    )
    parser.add_argument(
        "--stockfish-path",
        default=".local/data/stockfish/dataset",
        help="Path to stockfish dataset directory produced by save_to_disk().",
    )
    parser.add_argument(
        "--repo-id",
        default="kaupane/chess-positions",
        help="Target dataset repo, for example username/dataset-name.",
    )
    parser.add_argument(
        "--token", default=None, help="Optional Hugging Face token override."
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not exist.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="500MB",
        help="Maximum shard size passed to push_to_hub().",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Push even if the target repo already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from pondera.data.publish import run

    run(args)
