"""Encode position datasets for policy training."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", help="Hugging Face repository; alternative to local parquet files"
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--train-split", default="stockfish_train")
    parser.add_argument("--val-split", default="stockfish_val")
    parser.add_argument("--train-file", type=Path, action="append")
    parser.add_argument("--val-file", type=Path, action="append")
    parser.add_argument("--train-rows", type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path(".local/data/policy"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".local/data/hub"))
    args = parser.parse_args()
    if args.dataset and (args.train_file or args.val_file):
        parser.error("Choose a dataset repository or local files")
    if not args.dataset and not (args.train_file and args.val_file):
        parser.error("Provide --dataset or both --train-file and --val-file")
    from pondera.data.parquet import download_split
    from pondera.data.prepare import prepare_dataset

    train = args.train_file
    val = args.val_file
    if args.dataset:
        train = download_split(
            args.dataset, args.train_split, args.cache_dir, args.revision
        )
        val = download_split(
            args.dataset, args.val_split, args.cache_dir, args.revision
        )
    result = prepare_dataset(train, val, args.output_dir, args.train_rows, args.workers)
    print(json.dumps(result, indent=2))
