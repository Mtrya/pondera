"""Read Lichess puzzle Parquet and use the position after its setup move."""

import time
from collections import Counter
from pathlib import Path

import chess
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from pondera.data.roots.pool import _position_key, root_id_for, select_pool, split_for


def parquet_files(source, cache, *, local=None):
    if local is not None:
        local = Path(local)
        if local.is_file():
            return [local]
        files = sorted(local.rglob("*.parquet"))
    else:
        names = HfApi().list_repo_files(
            source.repo_id, repo_type="dataset", revision=source.revision
        )
        files = [
            Path(
                hf_hub_download(
                    source.repo_id,
                    name,
                    repo_type="dataset",
                    revision=source.revision,
                    cache_dir=cache,
                )
            )
            for name in sorted(names)
            if name.startswith("data/") and name.endswith(".parquet")
        ]
    if not files:
        raise FileNotFoundError(f"No puzzle Parquet files for {source.name}")
    return files


def puzzle_root(row, source):
    """One fresh root after Moves[0]; retain the complete source row."""
    board = chess.Board(row["FEN"])
    board.push_uci(row["Moves"].split()[0])
    if board.is_checkmate() or board.is_stalemate():
        return None
    fen = board.fen()
    key = _position_key(fen)
    return {
        "root_id": root_id_for(fen, 1),
        "split": split_for(key),
        "fen": fen,
        "repetition": 1,
        "turn": "white" if board.turn == chess.WHITE else "black",
        "position_key": key,
        "source": {
            "kind": "lichess_puzzles",
            "repo_id": source.repo_id,
            "revision": source.revision,
            "root_after_moves": 1,
            "repetition_is_assumed": True,
            "puzzle": row,
        },
    }


def build_pool(files, source, seed, *, exclude=()):
    started = time.monotonic()
    seen = set(exclude)
    counts = Counter()

    def records():
        for path in files:
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=source.batch_size, use_threads=False
            ):
                for row in batch.to_pylist():
                    counts["rows"] += 1
                    root = puzzle_root(row, source)
                    if root is None:
                        counts["terminal"] += 1
                        continue
                    key = root["position_key"]
                    if key in seen:
                        counts["duplicates_or_excluded"] += 1
                        continue
                    seen.add(key)
                    yield root

    # Puzzle rows have no full game history; do not impose a claimable quota.
    selected, strata = select_pool(records(), source.train, source.val, None, seed)
    return selected, {
        "kind": source.kind,
        "repo_id": source.repo_id,
        "revision": source.revision,
        "parameters": {"train": source.train, "val": source.val},
        "build": {**counts, "elapsed_seconds": time.monotonic() - started},
        "strata": strata,
    }
