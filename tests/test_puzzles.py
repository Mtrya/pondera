"""Genuine puzzle rows qualify source preparation and shared trace consumers."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tools.export_hf import export_hf
from trace_fixtures import trace_config_text

from pondera.chess.codec import encode_board
from pondera.chess.merge import MergeDictionary, decode
from pondera.chess.vocabulary import CHESS_VOCAB_SIZE
from pondera.data.config import load_config
from pondera.data.dataset import HF_COLUMNS, load_manifest, parquet_sources
from pondera.data.packing import PackedDocuments
from pondera.data.pipeline import validate_release
from pondera.data.roots import prepare_pool
from pondera.data.roots.pool import CorpusError, RootPool, read_roots
from pondera.data.tokenize import merge_and_encode
from pondera.evaluation.checkpoint import select_rows

FIXTURES = Path(__file__).parent / "fixtures" / "lichess_puzzles"


def source_rows():
    return [
        json.loads(line) for line in (FIXTURES / "rows.jsonl").read_text().splitlines()
    ]


class PuzzleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rows = source_rows()
        self.source = load_config().sources[1]

    def parquet(self, rows, name="puzzles.parquet"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
        return path

    def config(self, fraction=1.0):
        path = self.root / "data.toml"
        text = trace_config_text(width=1, depth=3, presentations={"full": 1.0})
        begin, end = text.index("[sources.elite]"), text.index("[sources.puzzles]")
        path.write_text(
            (text[:begin] + text[end:])
            .replace("validation_count = 2000", "validation_count = 0")
            .replace("prefix_fraction = 1.0", f"prefix_fraction = {fraction}")
            .replace("workers = 8", "workers = 1")
        )
        return path

    def test_cli_preserves_genuine_puzzle_rows_and_clocks(self):
        # 00008: ...Bxg3 captures the rook. 0000D: Qd6 is a quiet move.
        expected = {
            "00008": ("r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2b1/PqP3PP/7K w - - 0 25", "white"),
            "0000D": ("5rk1/1p3ppp/pq1Q1b2/8/8/1P3N2/P4PPP/3R2K1 b - - 3 27", "black"),
        }
        path = self.parquet(self.rows[:2])
        output = self.root / "pool"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pondera.data",
                "prepare",
                "--config",
                str(self.config()),
                "--input",
                f"puzzles={path}",
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        originals = {row["PuzzleId"]: row for row in self.rows[:2]}
        roots = list(RootPool(output).iter_range())
        self.assertEqual(len(roots), 2)
        for root in roots:
            source = root["source"]
            puzzle = source["puzzle"]
            self.assertEqual((root["fen"], root["turn"]), expected[puzzle["PuzzleId"]])
            self.assertEqual(root["repetition"], 1)
            self.assertEqual(puzzle, originals[puzzle["PuzzleId"]])
            self.assertEqual(source["root_after_moves"], 1)
            self.assertTrue(source["repetition_is_assumed"])
            self.assertIsNone(source["revision"])
        evaluated = select_rows(output, puzzle=False, seed=640, limit=2)
        self.assertEqual(
            {key for key, _ in evaluated}, {row["root_id"] for row in roots}
        )

    def test_prefix_dedup_and_validation_are_stable_across_batches_and_workers(self):
        archive = Path(__file__).parent / "fixtures" / "root_corpus" / "ties.pgn"
        elite, puzzles = load_config().sources
        elite = replace(
            elite,
            url=archive.resolve().as_uri(),
            sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        )
        first, second = self.rows[:2]
        collision = {
            **first,
            "PuzzleId": "elite-overlap",
            "FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 7 15",
            "Moves": "e2e4 e7e5",
        }
        terminal = {
            **first,
            "PuzzleId": "terminal",
            "FEN": "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
            "Moves": "f7g7",
        }
        self.parquet(
            [second, terminal, {**second, "Moves": "invalid"}], "puzzles/b.parquet"
        )
        self.parquet(
            [collision, first, {**first, "PuzzleId": "puzzle-overlap"}],
            "puzzles/a.parquet",
        )
        config = load_config()
        config = replace(
            config,
            sources=(elite, replace(puzzles, prefix_fraction=0.84, batch_size=1)),
            roots=replace(config.roots, validation_count=1, workers=1),
        )
        inputs = {"elite": archive, "puzzles": self.root / "puzzles"}
        output = self.root / "serial"
        manifest = prepare_pool(config, output, inputs=inputs)
        parallel_config = replace(
            config,
            roots=replace(config.roots, workers=4),
            sources=(replace(config.sources[1], batch_size=8), elite),
        )
        other = self.root / "parallel"
        parallel = prepare_pool(parallel_config, other, inputs=inputs)
        self.assertEqual(manifest["pool_id"], parallel["pool_id"])
        self.assertEqual(read_roots(output), read_roots(other))
        self.assertEqual(manifest["counts"], {"train": 3, "val": 1})
        self.assertEqual(manifest["source_order"], ["elite", "puzzles"])
        report = manifest["sources"]["puzzles"]
        self.assertEqual(
            tuple(
                report[field]
                for field in (
                    "raw_rows_available",
                    "raw_rows",
                    "terminal",
                    "duplicates_within_source",
                    "duplicates_elite",
                    "retained",
                )
            ),
            (6, 5, 1, 1, 1, 2),
        )
        roots = read_roots(output)
        # The frozen seed-640 ranks mix both sources and put 00008 on validation.
        self.assertEqual(
            [
                (
                    source["puzzle"]["PuzzleId"]
                    if "puzzle" in source
                    else source["ply"],
                    row["split"],
                )
                for row in roots
                for source in (row["source"],)
            ],
            [("0000D", "train"), ("00008", "val"), (2, "train"), (1, "train")],
        )
        e4 = roots[-1]
        self.assertEqual(
            e4["fen"], "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        )
        self.assertEqual(e4["source"]["game"], 0)

    def test_bad_puzzle_row_reports_original_file_and_ordinal_without_publication(self):
        first, second = self.rows[:2]
        path = self.parquet([first, {**second, "Moves": "e2e4"}])
        output = self.root / "invalid"
        with self.assertRaises(CorpusError) as error:
            prepare_pool(load_config(self.config()), output, inputs={"puzzles": path})
        self.assertIn(str(path), str(error.exception))
        self.assertIn("raw row 1", str(error.exception))
        self.assertFalse(output.exists())

    def test_remaining_eligible_roots_can_have_draw_claims(self):
        row = dict(self.rows[1])
        fields = row["FEN"].split()
        fields[4] = "99"
        row["FEN"] = " ".join(fields)
        output = self.root / "claims"
        prepare_pool(
            load_config(self.config()), output, inputs={"puzzles": self.parquet([row])}
        )
        [root] = read_roots(output)
        self.assertEqual(root["fen"].split()[4], "100")

    def test_puzzle_roots_curate_export_and_encode_with_the_real_engine(self):
        engine = os.environ.get("PONDERA_TRACE_ENGINE")
        if not engine or not Path(engine).is_file():
            self.skipTest("set PONDERA_TRACE_ENGINE to the instrumented engine")
        config_path = self.config()
        config = load_config(config_path)
        prepared = self.root / "prepared"
        prepare_pool(config, prepared, inputs={"puzzles": self.parquet(self.rows[:2])})
        before = (prepared / "roots.parquet").read_bytes()
        release = self.root / "release"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pondera.data",
                "curate",
                "--config",
                str(config_path),
                "--input",
                str(prepared),
                "--output",
                str(release),
                "--engine",
                engine,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual((prepared / "roots.parquet").read_bytes(), before)
        self.assertTrue(validate_release(release, workers=1)["ok"])
        exported = self.root / "shared"
        export_hf(release, exported, batch_size=1)
        training = self.root / "training"
        result = merge_and_encode(
            release,
            training,
            replace(
                config.merge, vocab_size=CHESS_VOCAB_SIZE + 8, min_count=1, batch_size=1
            ),
        )
        self.assertEqual(result["documents"], 2)
        dictionary = MergeDictionary(
            load_manifest(training / "packing.json")["merge_entries"]
        )
        rows = [
            row
            for path, _, _ in parquet_sources(release)
            for row in pq.read_table(path).to_pylist()
        ]
        self.assertEqual(
            [
                row
                for path, _, _ in parquet_sources(exported)
                for row in pq.read_table(path).to_pylist()
            ],
            [{column: row[column] for column in HF_COLUMNS} for row in rows],
        )
        documents = PackedDocuments(training)
        self.assertEqual(int((documents.split == 0).sum()), 2)
        for index, row in enumerate(rows):
            source = json.loads(row["source"])
            self.assertIn(source["puzzle"]["PuzzleId"], {"00008", "0000D"})
            self.assertEqual(source["repo_id"], "Lichess/chess-puzzles")
            self.assertEqual(
                decode(documents.document(index), dictionary).tolist(),
                encode_board(row["fen"], row["repetition"]).tolist()
                + row["trace_atom_ids"],
            )


if __name__ == "__main__":
    unittest.main()
