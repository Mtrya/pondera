"""Generation entry and graded puzzle evaluation on real models and fixtures."""

import json
import unittest
from pathlib import Path

import chess
import torch

from pondera.chess.vocabulary import CHESS_VOCAB_SIZE
from pondera.evaluation.puzzles import evaluate_puzzles, solve_puzzle
from pondera.inference.vanilla import choose_move
from pondera.models.vanilla import VanillaModel

FIXTURES = Path(__file__).parent / "fixtures" / "lichess_puzzles" / "rows.jsonl"


class GenerationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)
        self.model = VanillaModel(
            num_blocks=1,
            hidden_size=32,
            intermediate_size=48,
            num_heads=4,
            num_kv_heads=2,
            vocab_size=CHESS_VOCAB_SIZE,
        ).eval()

    def test_choose_move_is_legal_and_deterministic(self):
        rows = [json.loads(line) for line in FIXTURES.read_text().splitlines()[:3]]
        for row in rows:
            board = chess.Board(row["FEN"])
            board.push_uci(row["Moves"].split()[0])
            legal = {move.uci() for move in board.legal_moves}
            first = choose_move(self.model, board.fen())
            self.assertIn(first, legal)
            self.assertEqual(first, choose_move(self.model, board.fen()))

    def test_legal_move_mask_drives_the_choice(self):
        fen = chess.Board().fen()
        self.assertEqual(choose_move(self.model, fen, legal_moves=["e2e4"]), "e2e4")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_choose_move_uses_the_model_device(self):
        model = self.model.to("cuda")
        self.assertEqual(
            choose_move(model, chess.Board().fen(), legal_moves=["e2e4"]), "e2e4"
        )

    def test_puzzle_evaluation_counts_and_consistency(self):
        rows = [json.loads(line) for line in FIXTURES.read_text().splitlines()]
        report = evaluate_puzzles(lambda fen: choose_move(self.model, fen), rows)
        self.assertEqual(report["overall"]["count"], len(rows))
        self.assertLessEqual(report["overall"]["whole"], report["overall"]["first"])
        self.assertEqual(
            sum(bucket["count"] for bucket in report["buckets"].values()), len(rows)
        )

    def test_whole_line_requires_every_solver_move(self):
        row = json.loads(FIXTURES.read_text().splitlines()[0])
        correct = row["Moves"].split()[1]
        # The fixture line has three solver moves, so repeating the first
        # one is right on move one and wrong on move two.
        first, whole = solve_puzzle(lambda fen: correct, row)
        self.assertTrue(first)
        self.assertFalse(whole)
        first, whole = solve_puzzle(lambda fen: "a1a2", row)
        self.assertFalse(first)
        self.assertFalse(whole)


if __name__ == "__main__":
    unittest.main()
