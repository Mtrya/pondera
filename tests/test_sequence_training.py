import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import chess
import torch
from teacher_release import candidate_row, play, write_sequence_dataset

from pondera.data.trace import LEVEL0, LEVEL15
from pondera.models.checkpoint import read_checkpoint
from pondera.training.sequence import SequenceTrainConfig, train

RUY_LOPEZ = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]
EXCHANGE_START = "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"
EXCHANGE_MOVES = ["b1c3", "g8f6"]


def sequence_rows(records):
    rows = []
    for fen, _, _ in records:
        board = chess.Board(fen)
        moves = sorted(move.uci() for move in board.legal_moves)[:3]
        rows.append(candidate_row(fen, [(move, [move], None, 0.5) for move in moves]))
    return rows


class SequenceTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data = self.root / "data-level15"
        write_sequence_dataset(
            self.data,
            {
                "train": sequence_rows(play(RUY_LOPEZ)),
                "val": sequence_rows(play(EXCHANGE_MOVES, start=EXCHANGE_START)),
            },
            level=LEVEL15,
        )
        self.config = SequenceTrainConfig(
            model={
                "architecture": "sequence",
                "num_blocks": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
            },
            batch_size=2,
            accumulation_steps=2,
            epochs=2,
            workers=0,
            precision="fp32",
            log_every=1,
            val_every=2,
            save_every=1,
        )

    def final_metrics(self, output):
        rows = [
            json.loads(line)
            for line in (output / "metrics.jsonl").read_text().splitlines()
        ]
        return rows[-1]

    def test_training_records_trace_loss_and_final_move_top1(self):
        output = self.root / "run"
        train(self.config, self.data, output)
        final = self.final_metrics(output)
        self.assertTrue(final["training_complete"])
        self.assertGreater(final["val/loss"], 0.0)
        self.assertTrue(0.0 <= final["val/final_move_top1"] <= 1.0)
        logged = [
            json.loads(line)
            for line in (output / "metrics.jsonl").read_text().splitlines()
        ]
        losses = [row["train/loss"] for row in logged if "train/loss" in row]
        self.assertEqual(len(losses), 4)
        self.assertGreater(losses[0], 0.0)
        # Four updates at a warmup-peaked rate must reduce the training loss.
        self.assertLess(losses[-1], losses[0])
        checkpoint = read_checkpoint(output / "final.pth")
        self.assertEqual(checkpoint["model_config"]["architecture"], "sequence")
        # Eight rows, batch two, two accumulations, two epochs: four updates.
        self.assertEqual(checkpoint["step"], 4)
        self.assertEqual(checkpoint["examples_seen"], 16)

    def test_level0_dataset_trains(self):
        data = self.root / "data-level0"
        write_sequence_dataset(
            data,
            {
                "train": sequence_rows(play(RUY_LOPEZ)),
                "val": sequence_rows(play(EXCHANGE_MOVES, start=EXCHANGE_START)),
            },
            level=LEVEL0,
        )
        output = self.root / "run-level0"
        train(self.config, data, output)
        final = self.final_metrics(output)
        self.assertTrue(final["training_complete"])
        self.assertGreater(final["val/loss"], 0.0)
        self.assertEqual(final["val/rows"], 2)

    def test_resume_matches_uninterrupted_training(self):
        expected = read_checkpoint(train(self.config, self.data, self.root / "full"))
        output = self.root / "resumed"
        interrupted = train(self.config, self.data, output, max_steps=1)
        actual = read_checkpoint(
            train(self.config, self.data, output, resume=interrupted)
        )
        self.assertEqual(actual["step"], expected["step"])
        for key, value in expected["model_state_dict"].items():
            torch.testing.assert_close(
                actual["model_state_dict"][key], value, rtol=0, atol=0
            )

    def test_config_and_resume_contracts(self):
        with self.assertRaises(ValueError):
            replace(
                self.config,
                model={
                    "architecture": "base",
                    "num_blocks": 1,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_heads": 2,
                },
            )
        with self.assertRaises(ValueError):
            replace(self.config, val_every=0)
        output = self.root / "resume-check"
        interrupted = train(self.config, self.data, output, max_steps=1)
        with self.assertRaises(ValueError):
            train(
                replace(self.config, learning_rate=1e-3),
                self.data,
                output,
                resume=interrupted,
            )


if __name__ == "__main__":
    unittest.main()
