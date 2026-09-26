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
            self.assertTrue(unit["requested_depth_completed"])
            self.assertEqual(unit["incomplete_depths"], [])
            self.assertEqual(record["root_player"], "white")
            for snapshot in unit["per_depth"]:
                self.assertEqual(set(snapshot), {"d", "r1", "r2"})
                for rank in ["r1", "r2"]:
                    entry = snapshot[rank]
                    self.assertEqual(sum(entry["wdl"]), 1000)
                    self.assertAlmostEqual(
                        entry["value"], (entry["wdl"][0] - entry["wdl"][2]) / 1000
                    )
                    board = chess.Board()
                    for move in entry["pv"]:
                        self.assertIn(chess.Move.from_uci(move), board.legal_moves)
                        board.push_uci(move)
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

    def test_black_perspective_and_worker_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            records = [
                {"fen": chess.STARTING_FEN},
                {"fen": "6k1/8/8/8/8/8/5q2/7K b - - 0 1"},
            ]
            config = AnnotationConfig(depth=3, multipv=2)
            outputs = []
            for workers in [1, 2]:
                output = Path(directory) / f"workers-{workers}.jsonl"
                annotate_positions(records, output, config, workers)
                outputs.append(
                    [json.loads(line) for line in output.read_text().splitlines()]
                )
            self.assertEqual(
                [row["unit"]["per_depth"] for row in outputs[0]],
                [row["unit"]["per_depth"] for row in outputs[1]],
            )
            black = outputs[0][1]
            self.assertEqual(black["root_player"], "black")
            final = black["unit"]["per_depth"][-1]["r1"]
            self.assertGreater(final["value"], 0)
            if final["mate"] is not None:
                self.assertGreater(final["mate"], 0)
            else:
                self.assertGreater(final["cp"], 0)

    def test_multipv_is_capped_by_legal_moves_and_terminal_has_no_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            board = chess.Board("7k/8/5K2/8/8/8/8/1Q6 b - - 0 1")
            self.assertEqual(board.legal_moves.count(), 1)
            terminal = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
            self.assertTrue(terminal.is_checkmate())
            output = Path(directory) / "labels.jsonl"
            annotate_positions(
                [{"fen": board.fen()}, {"fen": terminal.fen()}],
                output,
                AnnotationConfig(depth=2, multipv=5),
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(rows[0]["unit"]["expected_candidates"], 1)
            self.assertTrue(rows[0]["unit"]["requested_depth_completed"])
            self.assertTrue(rows[1]["terminal"])
            self.assertNotIn("unit", rows[1])
