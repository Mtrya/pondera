import argparse
import collections
import csv
import itertools
import multiprocessing
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import chess
from datasets import (
    load_dataset,
    load_from_disk,
)
from tqdm import tqdm

from pondera.data.storage import (
    FINAL_FEATURES,
    CountBuckets,
    assemble_dataset,
    ensure_clean_output_dir,
    load_manifest,
    push_dataset,
    stable_hash_int,
    write_manifest,
)

SPLIT_HASH_DENOMINATOR = 1_000_000


@dataclass
class ExtractionBatchResult:
    games_processed: int
    positions_emitted: int
    aggregated_rows: List[Tuple[int, str, str, int]]


def pair_key(fen: str, next_move: str) -> str:
    return f"{fen}\t{next_move}"


def bucket_for_pair(fen: str, next_move: str, bucket_count: int) -> int:
    return stable_hash_int(pair_key(fen, next_move)) % bucket_count


def split_for_pair(fen: str, next_move: str, val_ratio: float) -> str:
    val_threshold = int(val_ratio * SPLIT_HASH_DENOMINATOR)
    hashed = stable_hash_int(pair_key(fen, next_move)) % SPLIT_HASH_DENOMINATOR
    return "validation" if hashed < val_threshold else "train"


def normalize_moves(raw_moves: object) -> List[str]:
    if isinstance(raw_moves, list):
        return [str(move) for move in raw_moves]
    if isinstance(raw_moves, str):
        return raw_moves.split()
    return []


def batched_games(
    dataset: Iterable[Dict],
    games_per_batch: int,
    skip_games: int = 0,
    max_games: Optional[int] = None,
) -> Iterator[List[List[str]]]:
    skipped = 0
    yielded = 0
    batch: List[List[str]] = []

    for sample in dataset:
        if skipped < skip_games:
            skipped += 1
            continue

        if max_games is not None and yielded >= max_games:
            break

        moves = normalize_moves(sample.get("moves"))
        if not moves:
            yielded += 1
            continue

        batch.append(moves)
        yielded += 1

        if len(batch) >= games_per_batch:
            yield batch
            batch = []

    if batch:
        yield batch


def extract_batch(
    game_batches: Sequence[List[str]], bucket_count: int
) -> ExtractionBatchResult:
    counts: collections.Counter[Tuple[str, str]] = collections.Counter()
    positions_emitted = 0

    for game_moves in game_batches:
        board = chess.Board()

        for move_uci in game_moves:
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                break

            if move not in board.legal_moves:
                break

            counts[(board.fen(), move_uci)] += 1
            positions_emitted += 1
            board.push(move)

    aggregated_rows = [
        (bucket_for_pair(fen, next_move, bucket_count), fen, next_move, count)
        for (fen, next_move), count in counts.items()
    ]

    return ExtractionBatchResult(
        games_processed=len(game_batches),
        positions_emitted=positions_emitted,
        aggregated_rows=aggregated_rows,
    )


def reduce_bucket_to_splits(
    bucket_path: Path,
    train_writer: csv.DictWriter,
    validation_writer: csv.DictWriter,
    val_ratio: float,
) -> Tuple[int, int]:
    conn = sqlite3.connect(f"file:{bucket_path}?mode=ro", uri=True)

    train_count = 0
    validation_count = 0
    try:
        cursor = conn.execute("SELECT fen, next_move, count FROM counts")

        for fen, next_move, count in cursor:
            row = {
                "fen": fen,
                "next_move": next_move,
                "count": int(count),
            }
            if split_for_pair(fen, next_move, val_ratio) == "validation":
                validation_writer.writerow(row)
                validation_count += 1
            else:
                train_writer.writerow(row)
                train_count += 1
    finally:
        conn.close()
    return train_count, validation_count


def run_extraction(
    dataset_name: str,
    split: str,
    output_dir: Path,
    num_workers: int,
    games_per_batch: int,
    bucket_count: int,
    skip_games: int,
    max_games: Optional[int],
    manifest_path: Path,
    manifest: Dict,
):
    extract_state = manifest.setdefault("extract", {})
    if extract_state.get("completed"):
        print("Extraction already completed; skipping.")
        return

    dataset = load_dataset(dataset_name, split=split)
    batch_iterator = batched_games(
        dataset=dataset,
        games_per_batch=games_per_batch,
        skip_games=skip_games,
        max_games=max_games,
    )

    bucket_dir = output_dir / "tmp" / "buckets"
    if bucket_dir.exists():
        shutil.rmtree(bucket_dir)
    writer = CountBuckets(bucket_dir, by_move=True)

    total_games = 0
    total_positions = 0
    total_aggregated_rows = 0

    pool = None
    if max_games is not None:
        progress_total: Optional[int] = max_games
    else:
        progress_total = max(0, len(dataset) - skip_games)
    pbar = tqdm(total=progress_total, desc="Extracting games", unit="game")

    try:
        if num_workers == 1:
            for batch in batch_iterator:
                result = extract_batch(batch, bucket_count)
                total_games += result.games_processed
                total_positions += result.positions_emitted
                total_aggregated_rows += writer.write_rows(result.aggregated_rows)
                result.aggregated_rows = []  # free the large list immediately
                pbar.update(result.games_processed)
                pbar.set_postfix(
                    {"pairs": total_positions, "rows": total_aggregated_rows}
                )
        else:
            pool = multiprocessing.Pool(processes=num_workers)
            # Feed the pool in bounded chunks so that Pool._handle_tasks does not
            # eagerly drain the entire generator into its unbounded internal task
            # queue, which would materialise every batch in memory at once.
            batch_args = ((batch, bucket_count) for batch in batch_iterator)
            chunk_size = num_workers * 2
            while True:
                chunk = list(itertools.islice(batch_args, chunk_size))
                if not chunk:
                    break
                for result in pool.imap_unordered(
                    _extract_batch_star, chunk, chunksize=1
                ):
                    total_games += result.games_processed
                    total_positions += result.positions_emitted
                    total_aggregated_rows += writer.write_rows(result.aggregated_rows)
                    result.aggregated_rows = []  # free the large list immediately
                    pbar.update(result.games_processed)
                    pbar.set_postfix(
                        {"pairs": total_positions, "rows": total_aggregated_rows}
                    )
    finally:
        pbar.close()
        writer.close()
        if pool is not None:
            pool.close()
            pool.join()

    extract_state.update(
        {
            "completed": True,
            "processed_games": total_games,
            "positions_emitted": total_positions,
            "aggregated_rows_merged": total_aggregated_rows,
            "bucket_count": bucket_count,
        }
    )
    write_manifest(manifest_path, manifest)


def _extract_batch_star(args) -> ExtractionBatchResult:
    return extract_batch(*args)


def run_reduction(
    output_dir: Path,
    val_ratio: float,
    manifest_path: Path,
    manifest: Dict,
):
    reduce_state = manifest.setdefault("reduce", {})
    if reduce_state.get("completed"):
        print("Reduction already completed; skipping.")
        return

    bucket_dir = output_dir / "tmp" / "buckets"
    if not bucket_dir.exists():
        raise FileNotFoundError(f"Missing bucket directory: {bucket_dir}")

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    train_csv = final_dir / "train.csv"
    validation_csv = final_dir / "validation.csv"

    with (
        train_csv.open("w", encoding="utf-8", newline="") as train_handle,
        validation_csv.open("w", encoding="utf-8", newline="") as validation_handle,
    ):
        train_writer = csv.DictWriter(
            train_handle, fieldnames=list(FINAL_FEATURES.keys())
        )
        validation_writer = csv.DictWriter(
            validation_handle, fieldnames=list(FINAL_FEATURES.keys())
        )
        train_writer.writeheader()
        validation_writer.writeheader()

        total_train_rows = 0
        total_validation_rows = 0
        bucket_paths = sorted(bucket_dir.glob("bucket-*.sqlite"))

        for bucket_path in tqdm(bucket_paths, desc="Reducing buckets", unit="bucket"):
            train_rows, validation_rows = reduce_bucket_to_splits(
                bucket_path=bucket_path,
                train_writer=train_writer,
                validation_writer=validation_writer,
                val_ratio=val_ratio,
            )
            total_train_rows += train_rows
            total_validation_rows += validation_rows

    reduce_state.update(
        {
            "completed": True,
            "train_rows": total_train_rows,
            "validation_rows": total_validation_rows,
            "val_ratio": val_ratio,
        }
    )
    write_manifest(manifest_path, manifest)

    if bucket_dir.exists():
        shutil.rmtree(bucket_dir)


def run_assembly(
    output_dir: Path,
    push_to_hub_flag: bool,
    repo_id: Optional[str],
    token: Optional[str],
    private: bool,
    manifest_path: Path,
    manifest: Dict,
):
    assemble_state = manifest.setdefault("assemble", {})
    dataset_dir = output_dir / "dataset"
    if assemble_state.get("completed") and not push_to_hub_flag:
        print("Dataset assembly already completed; skipping.")
        return

    final_dir = output_dir / "final"
    train_csv = final_dir / "train.csv"
    validation_csv = final_dir / "validation.csv"

    if not train_csv.exists() or not validation_csv.exists():
        raise FileNotFoundError("Missing final CSV splits. Run reduction first.")

    if assemble_state.get("completed") and dataset_dir.exists():
        dataset = load_from_disk(str(dataset_dir))
    else:
        dataset = assemble_dataset(train_csv, validation_csv, dataset_dir)

    assemble_state.update(
        {
            "completed": True,
            "dataset_dir": str(dataset_dir),
            "splits": {
                split_name: len(split_ds) for split_name, split_ds in dataset.items()
            },
        }
    )
    write_manifest(manifest_path, manifest)

    if push_to_hub_flag:
        if not repo_id:
            raise ValueError("--repo-id is required when --push-to-hub is set.")
        push_dataset(dataset=dataset, repo_id=repo_id, token=token, private=private)
        manifest["push"] = {
            "completed": True,
            "repo_id": repo_id,
            "private": private,
        }
        write_manifest(manifest_path, manifest)


def validate_args(args: argparse.Namespace):
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.games_per_batch < 1:
        raise ValueError("--games-per-batch must be >= 1")
    if args.bucket_count < 1:
        raise ValueError("--bucket-count must be >= 1")
    if args.skip_games < 0:
        raise ValueError("--skip-games must be >= 0")
    if args.max_games is not None and args.max_games < 1:
        raise ValueError("--max-games must be >= 1 when provided")
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must satisfy 0.0 <= val_ratio < 1.0")
    if args.push_to_hub and not args.repo_id:
        raise ValueError("--repo-id is required when --push-to-hub is set")


def run(args):
    validate_args(args)

    output_dir = Path(args.output_dir)
    ensure_clean_output_dir(output_dir, overwrite=args.overwrite)
    manifest_path = output_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    manifest["config"] = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "output_dir": str(output_dir),
        "num_workers": args.num_workers,
        "games_per_batch": args.games_per_batch,
        "bucket_count": args.bucket_count,
        "max_games": args.max_games,
        "skip_games": args.skip_games,
        "val_ratio": args.val_ratio,
    }
    write_manifest(manifest_path, manifest)

    run_extraction(
        dataset_name=args.dataset_name,
        split=args.split,
        output_dir=output_dir,
        num_workers=args.num_workers,
        games_per_batch=args.games_per_batch,
        bucket_count=args.bucket_count,
        skip_games=args.skip_games,
        max_games=args.max_games,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    run_reduction(
        output_dir=output_dir,
        val_ratio=args.val_ratio,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    run_assembly(
        output_dir=output_dir,
        push_to_hub_flag=args.push_to_hub,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        manifest_path=manifest_path,
        manifest=manifest,
    )
