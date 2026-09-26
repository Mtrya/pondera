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
    parser.add_argument(
        "--teacher-release", type=Path, help="Frozen annotation release manifest"
    )
    parser.add_argument("--teacher-depth", type=int, default=16)
    parser.add_argument("--train-split", default="stockfish_train")
    parser.add_argument("--val-split", default="stockfish_val")
    parser.add_argument("--train-file", type=Path, action="append")
    parser.add_argument("--val-file", type=Path, action="append")
    parser.add_argument("--train-rows", type=int)
    parser.add_argument("--val-rows", type=int)
    parser.add_argument("--tokenizer-version", type=int, choices=[1, 2], default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path(".local/data/policy"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".local/data/hub"))
    args = parser.parse_args()
    if args.teacher_release:
        if (
            args.dataset
            or args.train_file
            or args.val_file
            or args.train_rows
            or args.val_rows
        ):
            parser.error(
                "Teacher releases cannot be combined with dataset or row selection options"
            )
        if args.tokenizer_version != 2:
            parser.error("Teacher releases require the lossless tokenizer")
        from pondera.data.teacher import prepare_teacher_policy

        print(
            json.dumps(
                prepare_teacher_policy(
                    args.teacher_release, args.output_dir, args.teacher_depth
                ),
                indent=2,
            )
        )
        return
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
    result = prepare_dataset(
        train,
        val,
        args.output_dir,
        args.train_rows,
        args.workers,
        args.tokenizer_version,
        args.val_rows,
    )
    print(json.dumps(result, indent=2))
