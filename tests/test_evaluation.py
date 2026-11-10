import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import chess
import torch

from pondera.chess.vocabulary import CHESS_VOCAB_SIZE
from pondera.evaluation.match import parse_result, run_match
from pondera.models.vanilla import VanillaModel

CONFIG = {
    "num_blocks": 1,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_heads": 2,
    "num_kv_heads": 2,
}


def save_model(path):
    config = {**CONFIG, "vocab_size": CHESS_VOCAB_SIZE}
    torch.manual_seed(640)
    model = VanillaModel(**config)
    torch.save(
        {
            "format_version": 1,
            "arch": "vanilla",
            "model_config": config,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


class EvaluationTests(unittest.TestCase):
    def test_complete_and_incomplete_result(self):
        output = "Games: 2, Wins: 1, Losses: 0, Draws: 1\nElo: 190.85 +/- nan\n"
        result = parse_result(output, 2)
        self.assertEqual(result.score, 0.75)
        self.assertIsNone(result.elo_ci)
        with self.assertRaises(ValueError):
            parse_result(output, 4)

        all_loss = "Games: 2, Wins: 0, Losses: 2, Draws: 0\nElo: -inf +/- -nan\n"
        result = parse_result(all_loss, 2)
        self.assertEqual(
            (result.losses, result.elo_diff, result.elo_ci), (2, None, None)
        )

    def test_real_uci_process(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            save_model(checkpoint)
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pondera.inference.vanilla_uci",
                    "--checkpoint",
                    str(checkpoint),
                ],
                input="uci\nisready\nposition startpos moves e2e4\ngo\nquit\n",
                text=True,
                capture_output=True,
                check=True,
                timeout=30,
                env={**os.environ, "OMP_NUM_THREADS": "1"},
            )
            lines = process.stdout.splitlines()
            self.assertIn("uciok", lines)
            self.assertIn("readyok", lines)
            best = next(
                line.split()[1] for line in lines if line.startswith("bestmove ")
            )
            board = chess.Board()
            board.push_uci("e2e4")
            self.assertIn(chess.Move.from_uci(best), board.legal_moves)
            self.assertTrue(
                all(
                    line.startswith(("id ", "uciok", "readyok", "bestmove "))
                    for line in lines
                )
            )

    def test_uci_handshake_does_not_load_checkpoint(self):
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "pondera.inference.vanilla_uci",
                "--checkpoint",
                "/nonexistent/model.pth",
            ],
            input="uci\nisready\nquit\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        self.assertIn("readyok", process.stdout.splitlines())

    @unittest.skipUnless(
        shutil.which("fastchess") and shutil.which("stockfish"),
        "Requires fastchess and Stockfish on PATH",
    )
    def test_real_paired_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pth"
            save_model(checkpoint)
            openings = root / "openings.epd"
            openings.write_text("7k/5K2/6Q1/8/8/8/8/8 w - -\n")
            result = run_match(
                checkpoint, 1, root / "match", games=2, concurrency=1, openings=openings
            )
            self.assertEqual(result.games, 2)
            self.assertTrue((root / "match/result.json").is_file())
            self.assertTrue((root / "match/games.pgn").is_file())
