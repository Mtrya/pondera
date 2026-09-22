import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, load_from_disk

from pondera.chess.mapping import UCI_MOVE_TO_IDX
from pondera.cli.stockfish import parse_args
from pondera.data.human import (
    extract_batch,
    reduce_bucket_to_splits,
)
from pondera.data.positions import TokenizedPositions
from pondera.data.prepare import prepare_dataset
from pondera.data.stockfish import run
from pondera.data.storage import CountBuckets, assemble_dataset, load_manifest


class DataTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("stockfish"), "Requires Stockfish on PATH")
    def test_stockfish_dataset_pipeline_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            DatasetDict(
                {
                    "train": Dataset.from_list(
                        [
                            {
                                "fen": chess.STARTING_FEN,
                                "next_move": "e2e4",
                                "count": 2,
                            },
                            {
                                "fen": chess.STARTING_FEN,
                                "next_move": "d2d4",
                                "count": 3,
                            },
                        ]
                    )
                }
            ).save_to_disk(source)
            args = parse_args(
                [
                    "--source",
                    str(source),
                    "--depth",
                    "1",
                    "--num-workers",
                    "1",
                    "--bucket-count",
                    "2",
                    "--val-ratio",
                    "0",
                    "--output-dir",
                    str(root / "labeled"),
                ]
            )
            run(args)
            run(args)
            result = load_from_disk(root / "labeled/dataset")
            self.assertEqual(len(result["train"]), 1)
            self.assertEqual(result["train"][0]["count"], 5)
            self.assertIn(
                chess.Move.from_uci(result["train"][0]["next_move"]),
                chess.Board().legal_moves,
            )

    def test_prepare_from_real_parquet_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, move in enumerate(["e2e4", "d2d4"]):
                path = root / f"shard-{index}.parquet"
                pq.write_table(
                    pa.table(
                        {"fen": [chess.STARTING_FEN] * 3, "next_move": [move] * 3}
                    ),
                    path,
                )
                paths.append(path)
            output = root / "encoded"
            metadata = prepare_dataset(paths, paths, output, train_rows=4)
            dataset = TokenizedPositions(output, "train")
            self.assertEqual(len(dataset), 4)
            np.testing.assert_array_equal(
                dataset.moves,
                [UCI_MOVE_TO_IDX[m] for m in ["e2e4", "e2e4", "d2d4", "d2d4"]],
            )
            self.assertEqual(dataset.tokens.dtype, np.uint8)
            self.assertEqual(load_manifest(output / "manifest.json"), metadata)
            with self.assertRaises(FileExistsError):
                prepare_dataset(paths, paths, output)

    def test_human_extraction_sqlite_reduction_and_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = extract_batch([["e2e4", "e7e5"], ["e2e4"]], 1)
            manager = CountBuckets(root / "buckets", by_move=True)
            manager.write_rows(result.aggregated_rows)
            manager.close()
            train, val = root / "train.csv", root / "val.csv"
            with train.open("w", newline="") as t, val.open("w", newline="") as v:
                writers = [
                    csv.DictWriter(handle, ["fen", "next_move", "count"])
                    for handle in (t, v)
                ]
                for writer in writers:
                    writer.writeheader()
                counts = reduce_bucket_to_splits(
                    root / "buckets/bucket-00000.sqlite", *writers, 0.0
                )
            self.assertEqual(counts, (2, 0))
            dataset = assemble_dataset(train, val, root / "dataset")
            self.assertEqual(len(dataset["train"]), 2)
            self.assertEqual(len(dataset["validation"]), 0)
            self.assertEqual(sum(dataset["train"]["count"]), 3)
