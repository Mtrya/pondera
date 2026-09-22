"""Prepare memory-mapped policy targets from position parquet files."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pondera.chess.mapping import UCI_MOVE_TO_IDX
from pondera.chess.tokenize import TOKEN_COUNT, TOKENIZER_VERSION, encode_fens
from pondera.data.parquet import sampled_batches
from pondera.data.storage import write_manifest


def prepare_split(paths, output_dir: Path, split: str, rows=None, workers=0) -> int:
    total = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
    count = total if rows is None else rows
    if not 0 < count <= total:
        raise ValueError(f"Requested {count} rows from {total} available")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens = np.lib.format.open_memmap(
        output_dir / f"tokens_{split}.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(count, TOKEN_COUNT),
    )
    moves = np.lib.format.open_memmap(
        output_dir / f"moves_{split}.npy", mode="w+", dtype=np.int16, shape=(count,)
    )
    offset = 0
    for batch in sampled_batches(paths, ["fen", "next_move"], rows=count):
        fens = batch.column("fen").to_pylist()
        indices = [
            UCI_MOVE_TO_IDX[move] for move in batch.column("next_move").to_pylist()
        ]
        end = offset + len(fens)
        tokens[offset:end] = encode_fens(fens, num_workers=workers)
        moves[offset:end] = indices
        offset = end
    tokens.flush()
    moves.flush()
    return count


def prepare_dataset(train_paths, val_paths, output_dir, train_rows=None, workers=0):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    metadata = {
        "format_version": 1,
        "tokenizer_version": TOKENIZER_VERSION,
        "train_sources": [str(path) for path in train_paths],
        "val_sources": [str(path) for path in val_paths],
    }
    metadata["rows_train"] = prepare_split(
        train_paths, output_dir, "train", train_rows, workers
    )
    metadata["rows_val"] = prepare_split(val_paths, output_dir, "val", workers=workers)
    write_manifest(output_dir / "manifest.json", metadata)
    return metadata
