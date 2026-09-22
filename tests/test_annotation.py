import json
import shutil
import tempfile
import unittest
from pathlib import Path

import chess

from pondera.data.annotation import AnnotationConfig, annotate_positions


@unittest.skipUnless(shutil.which("stockfish"), "Requires Stockfish on PATH")
class AnnotationTests(unittest.TestCase):
    def test_real_annotation_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "labels.jsonl"
            records = [{"fen": chess.STARTING_FEN, "played": "e2e4", "count": 1}]
            config = AnnotationConfig(depth=2, multipv=2)
            self.assertEqual(annotate_positions(records, output, config), 1)
            record = json.loads(output.read_text())
            unit = record["unit"]
            self.assertEqual(unit["reached_depth"], 2)
            final = unit["per_depth"][-1]
            for rank in ["r1", "r2"]:
                board = chess.Board()
                self.assertEqual(final[rank]["bm"], final[rank]["pv"][0])
                for move in final[rank]["pv"]:
                    self.assertIn(chess.Move.from_uci(move), board.legal_moves)
                    board.push_uci(move)
            self.assertEqual(annotate_positions(records, output, config), 0)
            with self.assertRaises(ValueError):
                annotate_positions(records, output, AnnotationConfig(depth=3))
