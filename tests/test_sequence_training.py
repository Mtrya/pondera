import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import chess
import numpy as np
import torch
from teacher_release import play, sequence_row, write_sequence_dataset
from torch.nn import functional as F

from pondera.chess.vocabulary import CHESS_VOCAB_SIZE, FINAL, PAD
from pondera.data.positions import SequencePositions
from pondera.data.trace import (
    ROLE_CONTROL,
    ROLE_FINAL_MOVE,
    ROLE_PAD,
    ROLE_TOOL,
    TraceWriter,
)
from pondera.models.checkpoint import read_checkpoint
from pondera.models.factory import create_model
from pondera.training.cost import sequence_flops
from pondera.training.sequence import (
    SequenceCollator,
    SequenceTrainConfig,
    role_loss_totals,
    trace_loss,
    trace_loss_parts,
    train,
)

RUY_LOPEZ = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]
EXCHANGE_START = "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"
EXCHANGE_MOVES = ["b1c3", "g8f6"]
CASTLING_START = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"


def trace_row(fen, repetition=1):
    """One search-shaped trace: observe, walk, evaluate, return, decide."""
    board = chess.Board(fen)
    legal = sorted(move.uci() for move in board.legal_moves)
    writer = TraceWriter(fen, repetition)
    writer.board()
    writer.move(legal[0])
    writer.evaluate(value=0.25)
    writer.back()
    writer.final(legal[1])
    return sequence_row(fen, writer.tokens, writer.roles, repetition)


def direct_row(fen, repetition=1):
    board = chess.Board(fen)
    move = sorted(move.uci() for move in board.legal_moves)[0]
    writer = TraceWriter(fen, repetition)
    writer.final(move)
    return sequence_row(fen, writer.tokens, writer.roles, repetition)


def sequence_rows(records, build=trace_row):
    return [build(fen, repetition) for fen, _, repetition in records]


class SequenceTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data = self.root / "data-trace"
        write_sequence_dataset(
            self.data,
            {
                "train": sequence_rows(play(RUY_LOPEZ)),
                "val": sequence_rows(play(EXCHANGE_MOVES, start=EXCHANGE_START)),
            },
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

    def test_direct_answer_dataset_trains(self):
        data = self.root / "data-direct"
        write_sequence_dataset(
            data,
            {
                "train": sequence_rows(play(RUY_LOPEZ), build=direct_row),
                "val": sequence_rows(
                    play(EXCHANGE_MOVES, start=EXCHANGE_START), build=direct_row
                ),
            },
        )
        output = self.root / "run-direct"
        train(self.config, data, output)
        final = self.final_metrics(output)
        self.assertTrue(final["training_complete"])
        self.assertGreater(final["val/loss"], 0.0)
        self.assertEqual(final["val/rows"], 2)
        # A direct answer has no generated move or evaluation targets.
        self.assertNotIn("val/role/move_loss", final)
        self.assertNotIn("val/role/evaluation_loss", final)
        self.assertIn("val/role/control_loss", final)
        self.assertIn("val/role/final_loss", final)

    def test_trace_loss_weights_context_and_final_move_equally(self):
        row = SequencePositions(self.data, "val")[0]
        batch = SequenceCollator()([row])
        model = create_model(self.config.model)
        loss, logits, generated = trace_loss(model, batch)

        targets = batch["trace_ids"].long()
        ce = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)
        roles = batch["trace_roles"]
        context = (
            (roles != ROLE_PAD) & (roles != ROLE_TOOL) & (roles != ROLE_FINAL_MOVE)
        )
        final = roles == ROLE_FINAL_MOVE
        expected = (ce * context).sum() / context.sum()
        expected = expected + (ce * final).sum() / final.sum()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(generated, context | final)

        # Hand counts for this trace: BOARD request, move, evaluation, BACK,
        # and the FINAL delimiter form the context mean; one root move follows.
        self.assertEqual(int(context.sum()), 5)
        self.assertEqual(int(final.sum()), 1)
        # The observed start position is 32 piece atoms plus 8 metadata atoms.
        self.assertEqual(int((roles == ROLE_TOOL).sum()), 40)
        self.assertEqual(int((roles == ROLE_PAD).sum()), 0)
        # The BOARD request itself is supervised; its response is not.
        self.assertEqual(int(roles[0, 0]), ROLE_CONTROL)
        self.assertTrue(bool(generated[0, 0]))
        self.assertEqual(int(roles[0, 1]), ROLE_TOOL)
        self.assertFalse(bool(generated[0, 1]))

    def test_role_losses_partition_the_aggregate_loss(self):
        row = SequencePositions(self.data, "val")[0]
        batch = SequenceCollator()([row])
        model = create_model(self.config.model)
        loss, ce, _, _, _ = trace_loss_parts(model, batch)
        totals = role_loss_totals(ce, batch["trace_roles"])
        counts = {name: tokens for name, (_, tokens) in totals.items()}
        # This trace writes BOARD, one move, one evaluation, BACK, the FINAL
        # delimiter, and the final move. The 40-token BOARD response is tool
        # context and carries no role.
        self.assertEqual(counts, {"move": 1, "evaluation": 1, "control": 3, "final": 1})
        context = sum(
            float(totals[name][0].detach())
            for name in ("move", "evaluation", "control")
        )
        # Five context targets form one mean, plus the final-move mean.
        expected = torch.tensor(context / 5 + float(totals["final"][0].detach()))
        torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-6)

    def test_diagnostics_report_probe_roles_gaps_and_internals(self):
        config = replace(self.config, internals=True, internals_positions=2)
        output = self.root / "diagnostics"
        train(config, self.data, output)
        final = self.final_metrics(output)
        # The default train probe is larger than this fixture's train split.
        self.assertEqual(final["train_probe/rows"], 8)
        for prefix in ("val", "train_probe"):
            for role in ("move", "evaluation", "control", "final"):
                self.assertIn(f"{prefix}/role/{role}_loss", final)
                self.assertGreater(final[f"{prefix}/role/{role}_loss"], 0.0)
        self.assertNotIn("val/role/tool_loss", final)
        self.assertAlmostEqual(
            final["generalization/loss_gap"],
            final["val/loss"] - final["train_probe/loss"],
            places=6,
        )
        self.assertAlmostEqual(
            final["generalization/final_move_top1_gap"],
            final["val/final_move_top1"] - final["train_probe/final_move_top1"],
            places=6,
        )
        for key in (
            "internal/layer_00/residual_rms",
            "internal/layer_00/attention_entropy_mean",
            "internal/layer_00/effective_rank",
            "internal/layer_00/ffn_dead_fraction",
            "internal/embedding/tokens_starved_fraction",
            "internal/drift/embedding",
        ):
            self.assertIn(key, final)
        self.assertTrue(any((output / "probes").glob("internal-*.npz")))

    def test_flop_budget_stops_after_the_first_reaching_update(self):
        data = self.root / "data-pair"
        # Two board widths and two trace lengths in one optimizer batch.
        write_sequence_dataset(
            data,
            {
                "train": [trace_row(chess.STARTING_FEN), trace_row(CASTLING_START)],
                "val": sequence_rows(play(EXCHANGE_MOVES, start=EXCHANGE_START)),
            },
        )
        config = replace(
            self.config,
            batch_size=2,
            accumulation_steps=1,
            epochs=2,
            train_flops_budget=1.0,
            val_every=100,
        )
        output = self.root / "budget"
        train(config, data, output)
        final = self.final_metrics(output)
        self.assertEqual(final["step"], 1)
        self.assertTrue(final["flop_budget_reached"])
        self.assertTrue(final["training_complete"])
        self.assertEqual(final["train/examples"], 2)

        positions = SequencePositions(data, "train")
        lengths = [
            (
                int(np.count_nonzero(positions[index][0] != PAD)),
                int(positions[index][2]),
            )
            for index in range(len(positions))
        ]
        expected_board = sum(board for board, _ in lengths)
        expected_trace = sum(trace for _, trace in lengths)
        self.assertEqual(final["train/board_tokens"], expected_board)
        self.assertEqual(final["train/trace_tokens"], expected_trace)
        # Actual, not padded, lengths define the analytical budget.
        expected_flops = float(
            sequence_flops(
                self.config.model,
                [board for board, _ in lengths],
                [trace for _, trace in lengths],
            ).sum()
        )
        self.assertEqual(final["train/theoretical_flops"], expected_flops)
        self.assertGreater(final["train/theoretical_flops"], 1.0)

        checkpoint = read_checkpoint(output / "final.pth")
        self.assertEqual(checkpoint["step"], 1)
        self.assertEqual(checkpoint["train_flops"], final["train/theoretical_flops"])
        self.assertEqual(checkpoint["board_tokens"], expected_board)
        self.assertEqual(checkpoint["trace_tokens"], expected_trace)
        self.assertEqual(checkpoint["examples_seen"], 2)

    def test_resume_matches_uninterrupted_compute_budget_schedule(self):
        config = replace(self.config, train_flops_budget=1e12)
        expected = read_checkpoint(train(config, self.data, self.root / "full-budget"))
        output = self.root / "resumed-budget"
        interrupted = train(config, self.data, output, max_steps=1)
        actual = read_checkpoint(train(config, self.data, output, resume=interrupted))
        self.assertEqual(actual["step"], expected["step"])
        for key, value in expected["model_state_dict"].items():
            torch.testing.assert_close(
                actual["model_state_dict"][key], value, rtol=0, atol=0
            )
        for key in ("train_flops", "board_tokens", "trace_tokens", "examples_seen"):
            self.assertEqual(actual[key], expected[key])

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

    def test_collator_rejects_invalid_trace_structure(self):
        board, trace, length, roles = SequencePositions(self.data, "val")[0]
        cases = {}
        invalid = board.copy()
        invalid[0] = CHESS_VOCAB_SIZE
        cases["board vocabulary"] = (invalid, trace, length, roles)
        invalid = trace.copy()
        invalid[0] = CHESS_VOCAB_SIZE
        cases["trace vocabulary"] = (board, invalid, length, roles)
        invalid = trace.copy()
        invalid[-2] = invalid[-3]
        cases["missing FINAL"] = (board, invalid, length, roles)
        invalid = trace.copy()
        invalid[-1] = FINAL
        cases["final is not a move"] = (board, invalid, length, roles)
        cases["invalid length"] = (board, trace, len(trace) + 1, roles)
        for name, row in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                SequenceCollator()([row])

    def test_resume_rejects_changed_validation_manifest(self):
        output = self.root / "resume-validation-data"
        interrupted = train(self.config, self.data, output, max_steps=1)
        manifest_path = self.data / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["trace_tokens_val.npy"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            train(self.config, self.data, output, resume=interrupted)

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
        with self.assertRaises(ValueError):
            replace(self.config, train_probe_positions=-1)
        with self.assertRaises(ValueError):
            replace(self.config, internals_positions=0)
        with self.assertRaises(ValueError):
            replace(self.config, train_flops_budget=0.0)
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
