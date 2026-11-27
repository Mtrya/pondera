"""Read Lichess puzzle Parquet and use the position after its setup move."""

import time
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from pondera.chess import _rules as rules
from pondera.data.roots.pool import CorpusError, _position_key


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


def _setup_position(row):
    fen, moves = row["FEN"], row["Moves"]
    if not isinstance(fen, str) or len(fen.split()) != 6:
        raise ValueError("FEN must have six fields")
    if not isinstance(moves, str) or not (moves := moves.split(maxsplit=1)):
        raise ValueError("Moves must include the setup move")
    board = rules.Board(fen)
    if not board.is_valid():
        raise ValueError("Invalid setup position")
    board.push_uci(moves[0])
    if board.has_legal_moves():
        return _position_key(board.fen()), board.halfmove_clock, board.fullmove_number
    return None


def _cutoff(rows, fraction):
    return int(rows * Decimal(str(fraction)))


def collect(files, source, index, *, local=False):
    """Count the raw prefix before testing eligibility or deduplicating."""
    started = time.monotonic()
    total = 0
    for path in files:
        with pq.ParquetFile(path) as shard:
            total += shard.metadata.num_rows
    limit = _cutoff(total, source.prefix_fraction)
    counts = dict(
        raw_rows=0,
        terminal=0,
        duplicates_within_source=0,
        duplicates_elite=0,
        retained=0,
    )
    for path in files:
        if counts["raw_rows"] == limit:
            break
        ordinal = 0
        with pq.ParquetFile(path) as shard:
            for batch in shard.iter_batches(
                batch_size=source.batch_size, use_threads=False
            ):
                remaining = limit - counts["raw_rows"]
                for row in batch.slice(0, remaining).to_pylist():
                    try:
                        position = _setup_position(row)
                        outcome = (
                            "terminal"
                            if position is None
                            else index.add_puzzle(*position, row)
                        )
                    except (KeyError, TypeError, ValueError) as error:
                        raise CorpusError(
                            f"{path}: raw row {ordinal} (zero-based): {error}"
                        ) from error
                    counts[outcome] += 1
                    counts["raw_rows"] += 1
                    ordinal += 1
                if counts["raw_rows"] == limit:
                    break

    return {
        "kind": source.kind,
        "provenance": {
            "kind": source.kind,
            "repo_id": source.repo_id,
            "revision": None if local else source.revision,
            "root_after_moves": 1,
            "repetition_is_assumed": True,
        },
        "prefix_fraction": source.prefix_fraction,
        "raw_rows_available": total,
        **counts,
        "elapsed_seconds": time.monotonic() - started,
    }
