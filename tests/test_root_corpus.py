"""Prepare real PGN fixtures through the combined pool storage contract."""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

from pondera.chess import _rules as rules
from pondera.data.config import load_config
from pondera.data.roots import prepare_pool
from pondera.data.roots.pgn import ensure_archive
from pondera.data.roots.pool import CorpusError, RootPool, read_roots, validate_record

FIXTURES = Path(__file__).parent / "fixtures" / "root_corpus"
START_KEY = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
NF3_KEY = "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq -"
NF6_KEY = "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq -"
NG1_KEY = "rnbqkb1r/pppppppp/5n2/8/8/8/PPPPPPPP/RNBQKBNR b KQkq -"
E4_KEY = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"
E5_KEY = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"


def by_key(rows):
    return {row["position_key"]: row for row in rows}


class RootPoolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = load_config()

    def build(self, archive, validation=0, workers=1, limit_games=None):
        archive = FIXTURES / archive
        source = replace(
            self.config.sources[0],
            url=archive.resolve().as_uri(),
            limit_games=limit_games,
        )
        config = replace(
            self.config,
            sources=(source,),
            roots=replace(
                self.config.roots, validation_count=validation, workers=workers
            ),
        )
        output = self.root / f"{archive.stem}-{workers}"
        manifest = prepare_pool(config, output, inputs={"elite": archive})
        return RootPool(output), manifest["sources"]["elite"]

    def test_cli_publishes_actual_artifact_and_stable_identity(self):
        archive = FIXTURES / "ties.pgn"
        config = self.root / "recipe.toml"
        text = Path("configs/data.toml").read_text()
        a, b = text.index("[sources.puzzles]"), text.index("[search]")
        config.write_text(
            (text[:a] + text[b:])
            .replace("validation_count = 2000", "validation_count = 1")
            .replace(self.config.sources[0].url, archive.resolve().as_uri())
            .replace(
                self.config.sources[0].sha256,
                hashlib.sha256(archive.read_bytes()).hexdigest(),
            )
        )
        output = self.root / "prepared"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pondera.data",
                "prepare",
                "--config",
                str(config),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(result.stdout)
        pool = RootPool(report["root_pool"])
        self.assertEqual(Path(report["root_pool"]), output / "roots.parquet")
        self.assertEqual(report["roots"], 2)
        self.assertEqual(pool.pool_id, report["pool_id"])
        self.assertEqual(pool.manifest["counts"], {"train": 1, "val": 1})
        # Seed 640 selects e5 ahead of e4 in the validation hash order.
        self.assertEqual(
            [(row["position_key"], row["split"]) for row in pool.iter_range()],
            [(E5_KEY, "val"), (E4_KEY, "train")],
        )
        self.assertEqual(read_roots(output), list(pool.iter_range()))
        self.assertEqual(list(pool.iter_range(1, 1)), [])
        with self.assertRaises(ValueError):
            list(pool.iter_range(1, 3))

    def test_external_roots_reject_malformed_and_invalid_positions(self):
        for fen in (
            "broken/8/8/8/8/8/8/8 w - - 0 1",
            "rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ):
            with self.subTest(fen=fen), self.assertRaises(CorpusError):
                validate_record(
                    {
                        "root_id": "external",
                        "split": "train",
                        "fen": fen,
                        "repetition": 1,
                    }
                )

    def test_repetition_counts_match_the_hand_counted_game(self):
        pool, report = self.build("repetition.pgn")
        self.assertEqual(
            (report["games_read"], report["plies"], report["retained"]), (1, 8, 4)
        )
        roots = by_key(pool.iter_range())
        self.assertEqual(set(roots), {START_KEY, NF3_KEY, NF6_KEY, NG1_KEY})
        for key, clocks, repetition, ply in (
            (START_KEY, "8 5", 3, 8),
            (NF3_KEY, "5 3", 2, 5),
            (NF6_KEY, "6 4", 2, 6),
            (NG1_KEY, "7 4", 2, 7),
        ):
            row = roots[key]
            self.assertEqual(row["root_id"], f"{key} {clocks}|{repetition}")
            self.assertEqual(row["repetition"], repetition)
            self.assertEqual(row["source"]["game"], 0)
            self.assertEqual(row["source"]["ply"], ply)
            self.assertEqual(row["source"]["file"], "repetition.pgn")
            self.assertIsNone(row["source"]["archive_url"])

    def test_fivefold_last_visit_excludes_the_entire_per_game_key(self):
        pool, report = self.build("fivefold.pgn")
        self.assertEqual(report["fivefold"], 1)
        roots = by_key(pool.iter_range())
        self.assertNotIn(START_KEY, roots)
        self.assertEqual(
            [
                (roots[key]["repetition"], roots[key]["source"]["ply"])
                for key in (NF3_KEY, NF6_KEY, NG1_KEY)
            ],
            [(4, 13), (4, 14), (4, 15)],
        )

    def test_dedup_keeps_higher_repetition_and_first_game_on_ties(self):
        pool, report = self.build("dedup.pgn", validation=1)
        parallel, _ = self.build("dedup.pgn", validation=1, workers=4)
        rows = list(pool.iter_range())
        self.assertEqual(list(parallel.iter_range()), rows)
        self.assertEqual(list(RootPool(pool.path).iter_range(1, 3)), rows[1:3])
        self.assertEqual(report["duplicates_within_source"], 2)
        roots = by_key(rows)
        for key, repetition, game, ply in (
            (NF3_KEY, 2, 1, 5),
            (NF6_KEY, 1, 0, 2),
            (START_KEY, 2, 1, 4),
        ):
            row = roots[key]
            self.assertEqual(
                (row["repetition"], row["source"]["game"], row["source"]["ply"]),
                (repetition, game, ply),
            )

    def test_relocated_pool_keeps_its_identity_and_order(self):
        pool, _ = self.build("repetition.pgn", validation=1)
        destination = self.root / "relocated"
        shutil.copytree(pool.path.parent, destination)
        relocated = RootPool(destination)
        self.assertEqual(relocated.pool_id, pool.pool_id)
        self.assertEqual(list(relocated.iter_range()), list(pool.iter_range()))

    def test_comments_blank_lines_and_empty_games_preserve_archive_ordinals(self):
        archive = self.root / "boundaries.pgn"
        archive.write_text(
            '[Event "Empty"] [Variant "Chess960"]\n\n'
            '[Event "Aborted"] [SetUp "1"]\n\n{ no moves }\n'
            '[Event "First game"]\n\n'
            "1 . e4 { note\n  % evaluation } e5\n"
            "% ignored in column one\n\n2. Nf3 Nc6*\n\n"
            '[Event "Next game"]\n\n1. d4 d5 *\n'
        )
        pool, report = self.build(archive, workers=2)
        self.assertEqual(report["games_read"], 4)
        self.assertEqual(report["games_skipped"], {"setup": 1, "variant": 1})
        roots = by_key(pool.iter_range())
        self.assertEqual(
            sorted((r["source"]["game"], r["source"]["ply"]) for r in roots.values()),
            [(2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2)],
        )
        self.assertEqual(roots[E5_KEY]["source"]["ply"], 2)

    def test_invalid_games_keep_their_archive_ordinals(self):
        pool, report = self.build("corrupt.pgn")
        self.assertEqual(report["games_read"], 5)
        self.assertEqual(
            report["games_skipped"], {"illegal": 1, "setup": 1, "variant": 1}
        )
        rows = list(pool.iter_range())
        self.assertEqual(len(rows), 9)
        self.assertEqual({row["source"]["game"] for row in rows}, {1, 4})
        self.assertEqual(
            max(row["source"]["ply"] for row in rows if row["source"]["game"] == 4), 5
        )
        prefix, prefix_report = self.build("corrupt.pgn", limit_games=2, workers=2)
        self.assertEqual(prefix_report["games_read"], 2)
        self.assertEqual(len(prefix), 4)
        self.assertEqual({row["source"]["game"] for row in prefix.iter_range()}, {1})

    def test_terminal_positions_are_excluded(self):
        pool, report = self.build("terminal.pgn")
        self.assertEqual((report["plies"], report["terminal"], len(pool)), (26, 2, 24))
        for row in pool.iter_range():
            self.assertTrue(rules.Board(row["fen"]).has_legal_moves())

    def test_zip_archive_records_its_pgn_member(self):
        archive = self.root / "month.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.write(FIXTURES / "repetition.pgn", "month.pgn")
        pool, report = self.build(archive)
        self.assertEqual(report["provenance"]["file"], "month.pgn")
        self.assertEqual(len(pool), 4)
        with zipfile.ZipFile(archive, "a") as handle:
            handle.writestr("another.pgn", "")
        with self.assertRaises(CorpusError):
            self.build(archive, workers=2)

    @unittest.skipUnless(shutil.which("zstd"), "zstd binary is unavailable")
    def test_zstd_archive_can_be_read(self):
        archive = self.root / "month.zst"
        subprocess.run(
            ["zstd", "-q", str(FIXTURES / "repetition.pgn"), "-o", str(archive)],
            check=True,
        )
        pool, report = self.build(archive)
        self.assertEqual(report["provenance"]["file"], "month.zst")
        self.assertEqual(len(pool), 4)

    def test_archive_digest_is_checked_at_download(self):
        archive = self.root / "archive.pgn"
        contents = (FIXTURES / "repetition.pgn").read_bytes()
        archive.write_bytes(contents)
        source = replace(self.config.sources[0], url=archive.as_uri())
        cache = self.root / "cache"
        with self.assertRaises(CorpusError):
            ensure_archive(cache, source)
        verified = replace(source, sha256=hashlib.sha256(contents).hexdigest())
        downloaded = ensure_archive(cache, verified)
        self.assertEqual(downloaded.read_bytes(), contents)
        archive.unlink()
        self.assertEqual(ensure_archive(cache, verified), downloaded)

    def test_source_cache_and_pool_can_use_the_default_source_name(self):
        archive = FIXTURES / "ties.pgn"
        source = replace(
            self.config.sources[0],
            url=archive.as_uri(),
            sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        )
        config = replace(
            self.config,
            sources=(source,),
            roots=replace(self.config.roots, validation_count=1),
        )
        output = self.root / "source"
        with self.assertRaises(ValueError):
            prepare_pool(config, output, cache=output / "nested")
        self.assertFalse(output.exists())
        prepare_pool(config, output)
        self.assertEqual(len(RootPool(output)), 2)
