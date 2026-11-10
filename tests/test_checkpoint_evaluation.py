"""Live engine references and checkpoint evaluation on small real positions."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from pondera.chess.merge import MergeDictionary
from pondera.chess.vocabulary import (
    CHESS_VOCAB_SIZE,
    END,
    FINAL,
    MOVE,
    UCI_MOVE_TO_IDX,
    VALUE,
)
from pondera.evaluation.checkpoint import select_rows
from pondera.evaluation.puzzles import solve_puzzle
from pondera.evaluation.reference import Reference
from pondera.inference.trace import parse_answer
from pondera.models.vanilla import VanillaModel

FEN = "7k/5K2/6Q1/8/8/8/8/8 w - - 0 1"
PUZZLE = {
    "PuzzleId": "alternative-mates",
    "FEN": "8/5K1k/6Q1/8/8/8/8/8 b - - 0 1",
    "Moves": "h7h8 g6g7",
    "Rating": 800,
}


class PuzzleMateTests(unittest.TestCase):
    def test_any_legal_checkmate_solves_a_mate_in_one(self):
        board = chess.Board(FEN)
        for move in board.legal_moves:
            after = board.copy()
            after.push(move)
            if after.is_checkmate():
                self.assertEqual(
                    solve_puzzle(lambda fen: move.uci(), PUZZLE), (True, True)
                )


class SelectionTests(unittest.TestCase):
    def test_selection_is_invariant_to_input_order_duplicates_and_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            board = chess.Board()
            rows = [{"fen": board.fen(), "repetition": 1}]
            for move in ("e2e4", "e7e5", "g1f3"):
                board.push_uci(move)
                rows.append({"fen": board.fen(), "repetition": 1})
            forward, reverse = root / "forward.jsonl", root / "reverse.parquet"
            forward.write_text("".join(json.dumps(row) + "\n" for row in rows))
            pq.write_table(pa.Table.from_pylist(rows[::-1] + rows), reverse)
            one = select_rows(forward, puzzle=False, limit=2, seed=640)
            two = select_rows(reverse, puzzle=False, limit=2, seed=640)
            self.assertEqual(one, two)
            self.assertEqual(len(one), 2)


@unittest.skipUnless(shutil.which("stockfish"), "Requires Stockfish on PATH")
class ReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_mate_draw_root_value_perspective_and_cache_reuse(self):
        reference = Reference("stockfish", self.root / "scores.sqlite", nodes=2000)
        try:
            for board, mate, draw in (
                (chess.Board(FEN), "g6g7", "g6f5"),
                (chess.Board(FEN).mirror(), "g3g2", "g3f4"),
            ):
                # Root-relative move tokens are the same in the mirrored position.
                dictionary = MergeDictionary()
                best = parse_answer(
                    [FINAL, MOVE + UCI_MOVE_TO_IDX["g6g7"], VALUE + 19, END],
                    dictionary,
                    board.fen(),
                )
                self.assertEqual(best.decision, mate)
                result = reference.grade(board.fen(), 1, best)
                self.assertTrue(result["best_move"])
                self.assertEqual(result["root_value"], 1.0)
                self.assertEqual(result["value_absolute_error"], 0.0)
                after = board.copy()
                after.push_uci(draw)
                self.assertTrue(after.is_stalemate())
                wrong_move = parse_answer(
                    [FINAL, MOVE + UCI_MOVE_TO_IDX["g6f5"], VALUE + 19, END],
                    dictionary,
                    board.fen(),
                )
                result = reference.grade(board.fen(), 1, wrong_move)
                self.assertFalse(result["best_move"])
                self.assertGreater(result["move_rank"], 1)
                self.assertEqual(result["expected_score_loss"], 0.5)
                # Values target the root optimum even when the chosen move blunders.
                self.assertEqual(result["value_absolute_error"], 0.0)
            searches = reference.searches
            self.assertGreater(searches, 0)
        finally:
            reference.close()
        reference = Reference("stockfish", self.root / "scores.sqlite", nodes=2000)
        try:
            reference.scores(chess.Board(FEN))
            self.assertEqual(reference.searches, 0)
            self.assertEqual(reference.hits, 1)
        finally:
            reference.close()

    def test_pv_grades_each_sides_moves_from_that_sides_perspective(self):
        reference = Reference("stockfish", self.root / "pv.sqlite", nodes=1000)
        try:
            answer = parse_answer(
                [
                    FINAL,
                    MOVE + UCI_MOVE_TO_IDX["e2e4"],
                    VALUE + 10,
                    MOVE + UCI_MOVE_TO_IDX["e7e5"],
                    MOVE + UCI_MOVE_TO_IDX["g1f3"],
                    END,
                ],
                MergeDictionary(),
                chess.STARTING_FEN,
            )
            result = reference.grade(chess.STARTING_FEN, 1, answer, pv_plies=2)
            self.assertEqual([ply["move"] for ply in result["pv"]], ["e7e5", "g1f3"])
            self.assertEqual(
                [ply["root_player"] for ply in result["pv"]], [False, True]
            )
            self.assertTrue(all(ply["cp_loss"] >= 0 for ply in result["pv"]))
        finally:
            reference.close()

    def test_cli_records_failures_and_reuses_reference_across_checkpoints(self):
        config = dict(
            num_blocks=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            num_kv_heads=1,
            vocab_size=CHESS_VOCAB_SIZE,
        )
        model = VanillaModel(**config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        checkpoint = self.root / "model.pth"
        torch.save(
            {
                "arch": "vanilla",
                "model_config": config,
                "model_state_dict": model.state_dict(),
                "tokenizer_identity": {"merge_entries": []},
                "update": 7,
            },
            checkpoint,
        )
        roots = self.root / "roots.jsonl"
        roots.write_text(json.dumps({"fen": FEN, "repetition": 1}) + "\n")
        puzzles = self.root / "puzzles.jsonl"
        puzzles.write_text(json.dumps(PUZZLE) + "\n")
        recipe = self.root / "eval.toml"
        recipe.write_text(
            "[generation]\nreasoning_tokens=3\nanswer_tokens=4\n[engine]\nnodes=2000\npv_plies=0\n"
        )
        cache = self.root / "reference.sqlite"
        for index in range(2):
            output = self.root / f"evaluation-{index}"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pondera.evaluation",
                    "--checkpoint",
                    str(checkpoint),
                    "--config",
                    str(recipe),
                    "--roots",
                    str(roots),
                    "--puzzles",
                    str(puzzles),
                    "--output",
                    str(output),
                    "--cache",
                    str(cache),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "OMP_NUM_THREADS": "1"},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["checkpoint_metadata"]["update"], 7)
            self.assertEqual(set(report["budgets"]), {"0", "3"})
            for summary in report["budgets"].values():
                self.assertEqual(summary["overall"]["count"], 2)
                self.assertEqual(summary["overall"]["valid_rate"], 0)
                self.assertEqual(summary["overall"]["quality_count"], 0)
                self.assertIsNone(summary["root"]["cp_loss"]["mean"])
                self.assertEqual(summary["puzzle"]["first_accuracy"], 0)
                self.assertEqual(summary["puzzle"]["whole_accuracy"], 0)
            self.assertEqual(
                len((output / "positions.jsonl").read_text().splitlines()), 4
            )
            if index == 1:
                self.assertEqual(report["reference_cache"]["searches"], 0)


if __name__ == "__main__":
    unittest.main()
