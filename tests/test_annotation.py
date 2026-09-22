import json
import shutil
import tempfile
import unittest
from pathlib import Path

import chess

from pondera.data.annotation import AnnotationConfig, annotate_positions


@unittest.skipUnless(shutil.which("stockfish"), "Requires Stockfish on PATH")
class AnnotationTests(unittest.TestCase):
    def test_resume_recovers_unterminated_final_record(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "labels.jsonl"
            board = chess.Board()
            records = [{"fen": board.fen()}]
            board.push_uci("e2e4")
            records.append({"fen": board.fen()})
            config = AnnotationConfig(depth=2, multipv=1)
            annotate_positions(records[:1], output, config)
            committed = output.read_bytes()
            for tail in (
                b'{"fen": "',
                b'{"note": "\xe4\xb8',
                json.dumps(records[1]).encode(),
            ):
                with self.subTest(tail=tail):
                    output.write_bytes(committed + tail)
                    self.assertEqual(annotate_positions(records, output, config), 1)
                    self.assertTrue(output.read_bytes().startswith(committed))
                    self.assertEqual(
                        [
                            json.loads(line)["fen"]
                            for line in output.read_bytes().splitlines()
                        ],
                        [record["fen"] for record in records],
                    )

    def test_resume_rejects_malformed_complete_records(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "labels.jsonl"
            records = [{"fen": chess.STARTING_FEN}]
            config = AnnotationConfig(depth=2, multipv=1)
            annotate_positions(records, output, config)
            committed = output.read_bytes()
            for trailing in (b"", committed):
                with self.subTest(trailing=bool(trailing)):
                    corrupted = committed + b'{"fen":\n' + trailing
                    output.write_bytes(corrupted)
                    with self.assertRaises(json.JSONDecodeError):
                        annotate_positions(records, output, config)
                    self.assertEqual(output.read_bytes(), corrupted)

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
