"""Generation entry and graded puzzle evaluation on real models and fixtures."""

import json
import unittest
from pathlib import Path

import chess
import torch

from pondera.chess.vocabulary import CHESS_VOCAB_SIZE, MOVE, REPETITION, UCI_MOVE_TO_IDX
from pondera.evaluation.puzzles import evaluate_puzzles, solve_puzzle
from pondera.inference.vanilla import choose_move
from pondera.inference.vanilla_uci import pick
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

    def test_black_decisions_use_the_root_players_move_frame(self):
        with torch.no_grad():
            for parameter in self.model.parameters():
                if parameter.ndim == 2:
                    parameter.zero_()
            self.model.token_embedding.weight.fill_(1)
            self.model.head.weight[MOVE + UCI_MOVE_TO_IDX["e2e4"]].fill_(1)
        # Zero projections leave unit embeddings; only e2e4 has a positive logit.
        for turn, candidates, expected in (
            ("w", ["a2a3", "e2e4"], "e2e4"),
            ("b", ["a7a6", "e7e5"], "e7e5"),
        ):
            fen = f"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR {turn} KQkq - 0 1"
            self.assertEqual(
                choose_move(self.model, fen, legal_moves=candidates), expected
            )

    def test_uci_move_history_reaches_the_repetition_atom(self):
        inputs = []
        handle = self.model.token_embedding.register_forward_pre_hook(
            lambda module, args: inputs.append(args[0].tolist())
        )
        self.addCleanup(handle.remove)
        board = chess.Board()
        for count in range(1, 5):
            self.assertIn(
                chess.Move.from_uci(pick(self.model, board, "cpu")), board.legal_moves
            )
            self.assertEqual(inputs[-1][-2], REPETITION + count - 1)
            for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
                board.push_uci(move)

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
