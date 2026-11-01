import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

import chess

from pondera.data.root_corpus import (
    SOURCE_FILE,
    CorpusError,
    _Candidate,
    _profile,
    build_corpus,
    ensure_archive,
    split_for,
)
from pondera.data.storage import load_manifest, sha256
from pondera.data.trace_collect import _validated_root

FIXTURES = Path(__file__).parent / "fixtures" / "root_corpus"
START_KEY = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
NF3_KEY = "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq -"
NF6_KEY = "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq -"
NG1_KEY = "rnbqkb1r/pppppppp/5n2/8/8/8/PPPPPPPP/RNBQKBNR b KQkq -"
E4_KEY = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"
E5_KEY = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"


def build(fixture, directory, limit_games=None):
    archive = FIXTURES / fixture
    manifest = build_corpus(archive, directory, limit_games)
    path = Path(directory) / "roots.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return manifest, rows


def by_key(rows):
    return {row["position_key"]: row for row in rows}


def validate(rows):
    return [_validated_root(row, index) for index, row in enumerate(rows)]


class RootCorpusTests(unittest.TestCase):
    def test_repetition_counts_match_the_hand_counted_game(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, rows = build("repetition.pgn", directory)
        self.assertEqual(manifest["build"]["games_read"], 1)
        self.assertEqual(manifest["build"]["games_replayed"], 1)
        self.assertEqual(manifest["positions"]["occurrences"], 8)
        self.assertEqual(manifest["positions"]["unique_per_game"], 4)
        self.assertEqual(manifest["positions"]["pool_keys"], 4)
        roots = by_key(rows)
        self.assertEqual(set(roots), {START_KEY, NF3_KEY, NF6_KEY, NG1_KEY})
        expected = {
            START_KEY: (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 8 5",
                3,
                8,
                "white",
            ),
            NF3_KEY: (
                "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 5 3",
                2,
                5,
                "black",
            ),
            NF6_KEY: (
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 6 4",
                2,
                6,
                "white",
            ),
            NG1_KEY: (
                "rnbqkb1r/pppppppp/5n2/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 7 4",
                2,
                7,
                "black",
            ),
        }
        for key, (fen, repetition, ply, turn) in expected.items():
            root = roots[key]
            self.assertEqual(root["fen"], fen)
            self.assertEqual(root["repetition"], repetition)
            self.assertEqual(root["turn"], turn)
            self.assertEqual(
                root["source"], {"file": "repetition.pgn", "game": 0, "ply": ply}
            )
            self.assertEqual(
                root["root_id"],
                hashlib.sha256(f"{fen}|{repetition}".encode()).hexdigest(),
            )
        self.assertEqual(len({row["root_id"] for row in rows}), 4)
        self.assertEqual(len(validate(rows)), 4)

    def test_fivefold_repetition_is_not_emitted(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, rows = build("fivefold.pgn", directory)
        self.assertEqual(manifest["positions"]["occurrences"], 16)
        self.assertEqual(manifest["positions"]["unique_per_game"], 4)
        self.assertEqual(manifest["positions"]["pool_keys"], 3)
        self.assertEqual(manifest["positions"]["skipped_repetition_ge_5"], 1)
        self.assertNotIn(START_KEY, by_key(rows))
        self.assertEqual(sorted(row["repetition"] for row in rows), [4, 4, 4])
        self.assertEqual(
            [(row["position_key"], row["source"]["ply"]) for row in rows],
            [(NF3_KEY, 13), (NF6_KEY, 14), (NG1_KEY, 15)],
        )
        self.assertEqual(
            manifest["profile"]["game_records_by_repetition"], {"4": 3, "5+": 1}
        )

    def test_dedup_keeps_higher_repetition_then_first_arrival(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, rows = build("dedup.pgn", directory)
            tie_manifest, tie_rows = build("ties.pgn", directory)
        self.assertEqual(manifest["positions"]["replaced_by_higher_repetition"], 1)
        self.assertEqual(manifest["positions"]["duplicates_merged"], 1)
        roots = by_key(rows)
        self.assertEqual(roots[NF3_KEY]["repetition"], 2)
        self.assertEqual(
            roots[NF3_KEY]["source"], {"file": "dedup.pgn", "game": 1, "ply": 5}
        )
        self.assertEqual(roots[NF6_KEY]["repetition"], 1)
        self.assertEqual(
            roots[NF6_KEY]["source"], {"file": "dedup.pgn", "game": 0, "ply": 2}
        )
        self.assertEqual(roots[START_KEY]["repetition"], 2)
        self.assertEqual(
            roots[START_KEY]["source"], {"file": "dedup.pgn", "game": 1, "ply": 4}
        )
        tied = by_key(tie_rows)
        self.assertEqual(tie_manifest["positions"]["replaced_by_higher_repetition"], 0)
        self.assertEqual(tie_manifest["positions"]["duplicates_merged"], 2)
        self.assertEqual(
            tied[E4_KEY]["source"], {"file": "ties.pgn", "game": 0, "ply": 1}
        )
        self.assertEqual(
            tied[E5_KEY]["source"], {"file": "ties.pgn", "game": 0, "ply": 2}
        )

    def test_split_is_a_position_key_hash_bucket(self):
        self.assertEqual(split_for(START_KEY), "train")
        self.assertEqual(split_for(NF3_KEY), "train")
        self.assertEqual(
            split_for("rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq -"),
            "val",
        )
        keys = [f"position {index} w - -" for index in range(20000)]
        splits = [split_for(key) for key in keys]
        self.assertEqual(splits, [split_for(key) for key in keys])
        share = splits.count("val") / len(splits)
        self.assertAlmostEqual(share, 8192 / 532480, delta=0.003)

    def test_invalid_setup_variant_and_truncated_games_are_tolerated(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, rows = build("corrupt.pgn", directory)
        self.assertEqual(manifest["build"]["games_read"], 5)
        self.assertEqual(manifest["build"]["games_replayed"], 2)
        self.assertEqual(
            manifest["build"]["games_skipped"],
            {"illegal": 1, "setup": 1, "variant": 1},
        )
        self.assertEqual({row["source"]["game"] for row in rows}, {1, 4})
        self.assertEqual(
            max(row["source"]["ply"] for row in rows if row["source"]["game"] == 4), 5
        )
        self.assertEqual(len(validate(rows)), 9)

    def test_terminal_positions_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, rows = build("terminal.pgn", directory)
        self.assertEqual(manifest["build"]["games_replayed"], 2)
        self.assertEqual(manifest["positions"]["occurrences"], 26)
        self.assertEqual(manifest["positions"]["skipped_terminal"], 2)
        self.assertEqual(manifest["positions"]["pool_keys"], 24)
        for row in rows:
            board = chess.Board(row["fen"])
            self.assertFalse(board.is_checkmate())
            self.assertFalse(board.is_stalemate())
            self.assertTrue(board.legal_moves)
        self.assertNotIn(
            "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq -",
            by_key(rows),
        )
        self.assertNotIn(
            "5bnr/4p1pq/4Qpkr/7p/7P/4P3/PPPP1PP1/RNB1KBNR b KQ - 2 10",
            {row["fen"] for row in rows},
        )

    def test_manifest_records_identity_counts_and_profile(self):
        archive = FIXTURES / "repetition.pgn"
        with tempfile.TemporaryDirectory() as directory:
            build_corpus(archive, directory)
            manifest = load_manifest(Path(directory) / "roots.manifest.json")
            roots_path = Path(directory) / "roots.jsonl"
            self.assertEqual(manifest["kind"], "elite-root-corpus")
            self.assertEqual(manifest["format_version"], 1)
            self.assertEqual(manifest["source"]["sha256"], sha256(archive))
            self.assertEqual(manifest["source"]["bytes"], archive.stat().st_size)
            self.assertEqual(manifest["source"]["member"], "repetition.pgn")
            self.assertEqual(
                manifest["files"]["roots.jsonl"]["sha256"], sha256(roots_path)
            )
            self.assertEqual(manifest["files"]["roots.jsonl"]["rows"], 4)
            self.assertEqual(manifest["splits"], {"train": 4, "val": 0})
            profile = manifest["profile"]
            self.assertEqual(
                profile["repetition"]["counts"], {"1": 0, "2": 3, "3": 1, "4": 0}
            )
            self.assertEqual(profile["claimable"]["count"], 1)
            self.assertEqual(profile["claimable"]["repetition_ge_3"], 1)
            self.assertEqual(profile["claimable"]["halfmove_ge_100"], 0)
            self.assertEqual(profile["halfmove"]["counts"]["0-9"], 4)
            self.assertEqual(profile["fullmove_phase"]["counts"]["1-10"], 4)
            self.assertEqual(profile["positions_per_game"]["mean"], 8.0)
            self.assertEqual(profile["game_records_by_repetition"], {"2": 3, "3": 1})

    def test_claimable_covers_repetition_and_halfmove_thresholds(self):
        pool = {
            "a w - -": _Candidate(1, 99, 50, 0, 197),
            "b w - -": _Candidate(1, 100, 50, 0, 199),
            "c w - -": _Candidate(3, 10, 20, 0, 39),
            "d w - -": _Candidate(2, 10, 60, 0, 119),
        }
        profile = _profile(pool, Counter({1: 4, 2: 1}), [10, 20])
        self.assertEqual(profile["claimable"]["count"], 2)
        self.assertEqual(profile["claimable"]["repetition_ge_3"], 1)
        self.assertEqual(profile["claimable"]["halfmove_ge_100"], 1)
        self.assertEqual(profile["halfmove"]["counts"]["90-99"], 1)
        self.assertEqual(profile["halfmove"]["counts"]["100+"], 1)
        self.assertEqual(profile["fullmove_phase"]["counts"]["11-20"], 1)
        self.assertEqual(profile["fullmove_phase"]["counts"]["41-60"], 3)

    def test_zip_archive_is_streamed(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "month.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.write(FIXTURES / "repetition.pgn", "month.pgn")
            manifest = build_corpus(archive, Path(directory) / "out")
        self.assertEqual(manifest["source"]["member"], "month.pgn")
        self.assertEqual(manifest["files"]["roots.jsonl"]["rows"], 4)
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "empty.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("a.pgn", "")
                handle.writestr("b.pgn", "")
            with self.assertRaises(CorpusError):
                build_corpus(archive, Path(directory) / "out")

    @unittest.skipUnless(shutil.which("zstd"), "zstd binary is unavailable")
    def test_zst_archive_is_streamed_through_the_system_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "month.zst"
            subprocess.run(
                [
                    "zstd",
                    "-q",
                    "-f",
                    str(FIXTURES / "repetition.pgn"),
                    "-o",
                    str(archive),
                ],
                check=True,
            )
            manifest = build_corpus(archive, Path(directory) / "out")
        self.assertEqual(manifest["files"]["roots.jsonl"]["rows"], 4)
        self.assertEqual(manifest["positions"]["pool_keys"], 4)

    def test_download_rejects_an_archive_with_the_pinned_name_but_wrong_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / SOURCE_FILE
            path.write_bytes(b"not the pinned archive")
            with self.assertRaises(CorpusError):
                ensure_archive(directory)

    def test_limit_games_stops_at_the_input_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, rows = build("corrupt.pgn", directory, limit_games=2)
        self.assertEqual(manifest["build"]["games_read"], 2)
        self.assertEqual(manifest["build"]["games_replayed"], 1)
        self.assertEqual({row["source"]["game"] for row in rows}, {1})

    def test_manifest_position_accounting_closes(self):
        for fixture in (
            "repetition.pgn",
            "fivefold.pgn",
            "dedup.pgn",
            "corrupt.pgn",
            "terminal.pgn",
        ):
            with self.subTest(fixture=fixture):
                with tempfile.TemporaryDirectory() as directory:
                    manifest, rows = build(fixture, directory)
                positions = manifest["positions"]
                self.assertEqual(
                    positions["unique_per_game"],
                    positions["pool_keys"]
                    + positions["duplicates_merged"]
                    + positions["replaced_by_higher_repetition"]
                    + positions["skipped_repetition_ge_5"],
                )
                self.assertGreaterEqual(
                    positions["occurrences"],
                    positions["unique_per_game"] + positions["skipped_terminal"],
                )
                self.assertEqual(
                    sum(manifest["profile"]["repetition"]["counts"].values()), len(rows)
                )
                self.assertEqual(
                    sum(manifest["profile"]["halfmove"]["counts"].values()), len(rows)
                )
                self.assertEqual(
                    sum(manifest["profile"]["fullmove_phase"]["counts"].values()),
                    len(rows),
                )


if __name__ == "__main__":
    unittest.main()
