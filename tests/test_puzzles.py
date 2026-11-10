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

from pondera.chess.merge import MergeDictionary, decode
from pondera.chess.vocabulary import CHESS_VOCAB_SIZE
from pondera.data.config import load_config
from pondera.data.dataset import load_manifest, parquet_sources
from pondera.data.pipeline import validate_release
from pondera.data.roots import prepare_pool
from pondera.data.roots.pool import _claimable, read_roots
from pondera.data.roots.puzzles import build_pool, puzzle_root
from pondera.data.text import trace_atoms
from pondera.data.tokenize import merge_and_encode

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

    def parquet(self, rows):
        path = self.root / "puzzles.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
        return path

    def config(self, count):
        path = self.root / "data.toml"
        text = trace_config_text(width=1, depth=3, presentations={"full": 1.0})
        prefix, suffix = text.split("[sources.puzzles]", 1)
        prefix = prefix.replace("train = 500000", "train = 0").replace(
            "val = 2000", "val = 0"
        )
        suffix = suffix.replace("train = 0", f"train = {count}", 1)
        path.write_text(
            (prefix + "[sources.puzzles]" + suffix).replace(
                "workers = 8", "workers = 1"
            )
        )
        return path

    def test_setup_move_changes_the_board_and_turn_for_both_colors(self):
        # 00008: ...Bxg3 captures the rook. 0000D: Qd6 is a quiet move.
        expected = (
            ("r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2b1/PqP3PP/7K w - - 0 25", "white"),
            ("5rk1/1p3ppp/pq1Q1b2/8/8/1P3N2/P4PPP/3R2K1 b - - 3 27", "black"),
        )
        for row, (fen, turn) in zip(self.rows, expected):
            root = puzzle_root(row, self.source)
            self.assertEqual(root["fen"], fen)
            self.assertEqual(root["turn"], turn)
            self.assertEqual(root["repetition"], 1)
            self.assertEqual(root["source"]["puzzle"], row)
            self.assertEqual(root["source"]["root_after_moves"], 1)
            self.assertTrue(root["source"]["repetition_is_assumed"])

    def test_preparation_requires_an_enabled_source(self):
        config = load_config(self.config(0))
        with self.assertRaises(ValueError):
            prepare_pool(config, self.root / "disabled")

    def test_sampling_is_stable_across_batches_and_excludes_duplicate_positions(self):
        path = self.parquet(self.rows + self.rows[:1])
        source = replace(self.source, train=4, val=0, batch_size=1)
        first, report = build_pool([path], source, 640)
        second, _ = build_pool([path], replace(source, batch_size=7), 640)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(report["build"]["duplicates_or_excluded"], 1)
        excluded = {root["position_key"] for root in first[:2]}
        changed, report = build_pool([path], source, 640, exclude=excluded)
        self.assertEqual(len(changed), 4)
        self.assertFalse(excluded & {root["position_key"] for root in changed})
        self.assertEqual(len({root["position_key"] for root in changed}), 4)

    def test_puzzles_do_not_require_a_claimable_quota_or_exclude_rare_claims(self):
        # Keep the legal Qd6 setup but put its halfmove clock at the claim boundary.
        row = dict(self.rows[1])
        fields = row["FEN"].split()
        fields[4] = "99"
        row["FEN"] = " ".join(fields)
        source = replace(self.source, train=1, val=0)
        selected, _ = build_pool([self.parquet([row])], source, 640)
        self.assertEqual(len(selected), 1)
        self.assertTrue(_claimable(selected[0]))

    def test_preparation_uses_source_counts_and_keeps_complete_puzzle_provenance(self):
        path = self.parquet(self.rows)
        config_path = self.config(4)
        output = self.root / "prepared"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pondera.data",
                "prepare",
                "--config",
                str(config_path),
                "--input",
                f"puzzles={path}",
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        roots = read_roots(output / "roots.jsonl")
        self.assertEqual(len(roots), 4)
        originals = {row["PuzzleId"]: row for row in self.rows}
        for root in roots:
            source = root["source"]
            self.assertEqual(source["puzzle"], originals[source["puzzle"]["PuzzleId"]])
            self.assertIsNone(source["revision"])
            self.assertNotIn("archive_url", source)
        manifest = load_manifest(output / "pool.manifest.json")
        self.assertEqual(manifest["counts"], {"train": 4, "val": 0})
        self.assertEqual(set(manifest["sources"]), {"puzzles"})
        self.assertFalse((output / "source").exists())
        duplicate_output = self.root / "prepared-again"
        prepare_pool(
            load_config(config_path), duplicate_output, inputs={"puzzles": path}
        )
        self.assertEqual(
            (output / "roots.jsonl").read_bytes(),
            (duplicate_output / "roots.jsonl").read_bytes(),
        )

    def test_named_source_quotas_prepare_a_mixed_unique_pool(self):
        config_path = self.config(4)
        archive = Path(__file__).parent / "fixtures" / "root_corpus" / "ties.pgn"
        elite = load_config().sources[0]
        text = config_path.read_text()
        prefix, suffix = text.split("[sources.puzzles]", 1)
        prefix = prefix.replace("train = 0", "train = 2", 1).replace(
            "claimable_share = 0.03", "claimable_share = 0.0"
        )
        prefix = prefix.replace(elite.url, archive.resolve().as_uri()).replace(
            elite.sha256, hashlib.sha256(archive.read_bytes()).hexdigest()
        )
        config_path.write_text(prefix + "[sources.puzzles]" + suffix)
        output = self.root / "mixed"
        manifest = prepare_pool(
            load_config(config_path),
            output,
            inputs={"puzzles": self.parquet(self.rows)},
        )
        roots = read_roots(output / "roots.jsonl")
        self.assertEqual(manifest["source_order"], ["elite", "puzzles"])
        self.assertEqual(manifest["counts"], {"train": 6, "val": 0})
        self.assertEqual(len({root["position_key"] for root in roots}), 6)
        self.assertEqual(
            sum(root["source"].get("kind") == "lichess_puzzles" for root in roots), 4
        )
        self.assertEqual(sum("archive_url" in root["source"] for root in roots), 2)

    def test_puzzle_roots_curate_export_and_encode_with_the_real_engine(self):
        engine = os.environ.get("PONDERA_TRACE_ENGINE")
        if not engine or not Path(engine).is_file():
            self.skipTest("set PONDERA_TRACE_ENGINE to the instrumented engine")
        config_path = self.config(2)
        config = load_config(config_path)
        prepared = self.root / "prepared"
        prepare_pool(config, prepared, inputs={"puzzles": self.parquet(self.rows[:2])})
        before = (prepared / "roots.jsonl").read_bytes()
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
        self.assertEqual((prepared / "roots.jsonl").read_bytes(), before)
        self.assertTrue(validate_release(release, workers=1)["ok"])
        exported = self.root / "shared"
        export_hf(release, exported, batch_size=1)
        training = self.root / "training"
        result = merge_and_encode(
            exported,
            training,
            replace(
                config.merge, vocab_size=CHESS_VOCAB_SIZE + 8, min_count=1, batch_size=1
            ),
        )
        self.assertEqual(result["rows"], 2)
        dictionary = MergeDictionary(
            load_manifest(training / "release.json")["merge_entries"]
        )
        rows = [
            row
            for path, _, _ in parquet_sources(training)
            for row in pq.read_table(path).to_pylist()
        ]
        self.assertEqual(sum(row["split"] == "train" for row in rows), 2)
        for row in rows:
            source = json.loads(row["source"])
            self.assertIn(source["puzzle"]["PuzzleId"], {"00008", "0000D"})
            self.assertEqual(source["repo_id"], "Lichess/chess-puzzles")
            self.assertEqual(
                decode(row["trace_token_ids"], dictionary).tolist(),
                trace_atoms(row["trace_text"]).tolist(),
            )


if __name__ == "__main__":
    unittest.main()
