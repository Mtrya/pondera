"""Prepare memory-mapped policy targets from position parquet files."""

from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pondera.chess.board_tokens import BOARD_TOKENIZER_VERSION, PAD, encode_boards
from pondera.chess.mapping import UCI_MOVE_TO_IDX
from pondera.chess.tokenize import TOKEN_COUNT, TOKENIZER_VERSION, encode_fens
from pondera.data.parquet import sampled_batches
from pondera.data.storage import write_manifest


def prepare_split(
    paths,
    output_dir: Path,
    split: str,
    rows=None,
    workers=0,
    tokenizer_version=TOKENIZER_VERSION,
) -> int:
    total = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
    count = total if rows is None else rows
    if not 0 < count <= total:
        raise ValueError(f"Requested {count} rows from {total} available")
    schemas = [set(pq.ParquetFile(path).schema_arrow.names) for path in paths]
    has_repetition = ["repetition" in schema for schema in schemas]
    if any(has_repetition) and not all(has_repetition):
        raise ValueError(
            "All shards must agree on whether repetition counts are supplied"
        )
    columns = ["fen", "next_move"] + (["repetition"] if all(has_repetition) else [])
    width, dtype = TOKEN_COUNT, np.uint8
    if tokenizer_version == BOARD_TOKENIZER_VERSION:
        width, dtype = 0, np.int16
        for batch in sampled_batches(paths, columns, rows=count):
            repetitions = (
                batch.column("repetition").to_pylist()
                if all(has_repetition)
                else [1] * len(batch)
            )
            for fen, rep in zip(batch.column("fen").to_pylist(), repetitions):
                counters = [int(v) for v in fen.split()[-2:]] + [rep]
                width = max(
                    width, 76 + sum(max(1, (v.bit_length() + 7) // 8) for v in counters)
                )
    elif tokenizer_version != TOKENIZER_VERSION:
        raise ValueError(f"Unsupported tokenizer version: {tokenizer_version}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens = np.lib.format.open_memmap(
        output_dir / f"tokens_{split}.npy",
        mode="w+",
        dtype=dtype,
        shape=(count, width),
    )
    moves = np.lib.format.open_memmap(
        output_dir / f"moves_{split}.npy", mode="w+", dtype=np.int16, shape=(count,)
    )
    offset = 0
    for batch in sampled_batches(paths, columns, rows=count):
        fens = batch.column("fen").to_pylist()
        indices = [
            UCI_MOVE_TO_IDX[move] for move in batch.column("next_move").to_pylist()
        ]
        end = offset + len(fens)
        repetitions = (
            batch.column("repetition").to_pylist() if all(has_repetition) else None
        )
        if tokenizer_version == BOARD_TOKENIZER_VERSION:
            encoded = encode_boards(fens, repetitions)
            tokens[offset:end] = PAD
            tokens[offset:end, : encoded.shape[1]] = encoded
        else:
            tokens[offset:end] = encode_fens(fens, repetitions, num_workers=workers)
        moves[offset:end] = indices
        offset = end
    tokens.flush()
    moves.flush()
    return count


def prepare_dataset(
    train_paths,
    val_paths,
    output_dir,
    train_rows=None,
    workers=0,
    tokenizer_version=TOKENIZER_VERSION,
    val_rows=None,
):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    metadata = {
        "format_version": 1,
        "tokenizer_version": tokenizer_version,
        "train_sources": [str(path) for path in train_paths],
        "val_sources": [str(path) for path in val_paths],
    }
    metadata["rows_train"] = prepare_split(
        train_paths, output_dir, "train", train_rows, workers, tokenizer_version
    )
    metadata["rows_val"] = prepare_split(
        val_paths, output_dir, "val", val_rows, workers, tokenizer_version
    )
    metadata["files"] = {}
    for path in sorted(output_dir.glob("*.npy")):
        checksum = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                checksum.update(chunk)
        metadata["files"][path.name] = checksum.hexdigest()
    write_manifest(output_dir / "manifest.json", metadata)
    return metadata
