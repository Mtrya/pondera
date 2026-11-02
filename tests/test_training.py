import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import chess
import torch
from dataset_fixtures import encoded_row as dataset_row
from dataset_fixtures import play, write_trace_dataset
from torch.nn import functional as F

from pondera.data.dataset import TracePositions
from pondera.data.trace import (
    ROLE_BOARD,
    ROLE_PAD,
    TraceWriter,
)
from pondera.models.checkpoint import read_checkpoint
from pondera.models.policy import create_model
from pondera.training.trace import (
    TraceCollator,
    TrainConfig,
    role_loss_totals,
    trace_loss_parts,
    train,
)

RUY_LOPEZ = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]
EXCHANGE_START = "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"
EXCHANGE_MOVES = ["b1c3", "g8f6"]


def trace_row(fen, repetition=1):
    board = chess.Board(fen)
    legal = sorted(move.uci() for move in board.legal_moves)
    writer = TraceWriter(fen, repetition)
    writer.depth()
    writer.move(legal[0])
    writer.board()
    writer.evaluate(value=0.25)
    writer.back()
    writer.move(legal[1])
    writer.refuted()
    writer.back()
    writer.evaluate(value=0.25)
    writer.final(legal[2])
    writer.final_evaluate(value=0.25)
    after = chess.Board(fen)
    after.push_uci(legal[2])
    writer.final_pv(sorted(move.uci() for move in after.legal_moves)[0])
    writer.end()
    return dataset_row(fen, writer.tokens, writer.roles, repetition)


def direct_row(fen, repetition=1):
    board = chess.Board(fen)
    move = sorted(move.uci() for move in board.legal_moves)[0]
    writer = TraceWriter(fen, repetition)
    writer.final(move)
    writer.final_evaluate(value=0.0)
    writer.end()
    return dataset_row(fen, writer.tokens, writer.roles, repetition)


def trace_rows(records, build=trace_row):
    return [build(fen, repetition) for fen, _, repetition in records]


class TraceTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data = self.root / "data-trace"
        write_trace_dataset(
            self.data,
            {
                "train": trace_rows(play(RUY_LOPEZ)),
                "val": trace_rows(play(EXCHANGE_MOVES, start=EXCHANGE_START)),
            },
        )
        self.config = TrainConfig(
            model={
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
        self.assertLess(losses[-1], losses[0])
        checkpoint = read_checkpoint(output / "final.pth")
        self.assertEqual(checkpoint["step"], 4)
        self.assertEqual(checkpoint["examples_seen"], 16)

    def test_microbatch_regrouping_keeps_the_token_mean_objective(self):
        full = replace(
            self.config,
            batch_size=2,
            accumulation_steps=1,
            epochs=1,
            val_every=100,
            save_every=100,
        )
        split = replace(full, batch_size=1, accumulation_steps=2)
        first = read_checkpoint(train(full, self.data, self.root / "regroup-full"))
        second = read_checkpoint(train(split, self.data, self.root / "regroup-split"))
        difference = max(
            float((first["model_state_dict"][name] - value).abs().max())
            for name, value in second["model_state_dict"].items()
        )
        self.assertLess(difference, 1e-5)

    def test_direct_answer_dataset_trains(self):
        data = self.root / "data-direct"
        write_trace_dataset(
            data,
            {
                "train": trace_rows(play(RUY_LOPEZ), build=direct_row),
                "val": trace_rows(
                    play(EXCHANGE_MOVES, start=EXCHANGE_START), build=direct_row
                ),
            },
        )
        output = self.root / "run-direct"
        train(self.config, data, output)
        final = self.final_metrics(output)
        self.assertTrue(final["training_complete"])
        self.assertEqual(final["val/rows"], 2)
        for role in ("move", "evaluation", "refuted", "final_line"):
            self.assertNotIn(f"val/role/{role}_loss", final)
        self.assertIn("val/role/final_move_loss", final)

    def test_trace_loss_supervises_every_non_padding_token(self):
        row = TracePositions(self.data, "val")[0]
        batch = TraceCollator()([row])
        model = create_model(self.config.model)
        loss, ce, logits, generated = trace_loss_parts(model, batch)
        targets = batch["trace_ids"].long()
        expected_ce = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)
        expected_mask = batch["trace_roles"] != ROLE_PAD
        torch.testing.assert_close(ce, expected_ce)
        torch.testing.assert_close(generated, expected_mask)
        torch.testing.assert_close(loss, expected_ce[expected_mask].mean())

        # Hand count: 14 non-board tokens plus 32 piece atoms and 8 metadata atoms.
        self.assertEqual(int(generated.sum()), 54)
        self.assertEqual(int((batch["trace_roles"] == ROLE_BOARD).sum()), 40)
        self.assertTrue(bool(generated[0, 3]))
        totals = role_loss_totals(ce, batch["trace_roles"])
        total = sum(float(role_total.detach()) for role_total, _ in totals.values())
        tokens = sum(count for _, count in totals.values())
        self.assertEqual(tokens, 54)
        torch.testing.assert_close(loss, torch.tensor(total / tokens))

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
        for key in ("board_tokens", "trace_tokens", "examples_seen"):
            self.assertEqual(actual[key], expected[key])


if __name__ == "__main__":
    unittest.main()
