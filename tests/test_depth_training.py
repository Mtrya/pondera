import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from teacher_release import depth_row, play, write_depth_release

from pondera.data.teacher import prepare_teacher_depth
from pondera.models.checkpoint import read_checkpoint
from pondera.training.depth import DepthTrainConfig, train

RUY_LOPEZ = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]
EXCHANGE_START = "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"
EXCHANGE_MOVES = ["b1c3", "g8f6"]


def depth_line(moves, start=None):
    records = play(moves) if start is None else play(moves, start=start)
    return [
        depth_row(fen, [(1, move, None, 0.0), (2, move, None, 0.5)])
        for fen, move, _ in records
    ]


class DepthTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data = self.root / "data"
        release = write_depth_release(
            self.root,
            {
                "train": depth_line(RUY_LOPEZ),
                "val": depth_line(EXCHANGE_MOVES, start=EXCHANGE_START),
            },
        )
        prepare_teacher_depth(release, self.data, max_depth=2)
        self.config = DepthTrainConfig(
            model={
                "architecture": "depth",
                "cores": [{"layers": 1, "repetition": {"max_repetitions": 2}}],
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
            },
            teacher_depth=2,
            batch_size=2,
            accumulation_steps=2,
            epochs=2,
            workers=0,
            precision="fp32",
            log_every=1,
            val_every=2,
            save_every=1,
            diagnostics=True,
        )

    def final_metrics(self, output):
        rows = [
            json.loads(line)
            for line in (output / "metrics.jsonl").read_text().splitlines()
        ]
        return rows[-1]

    def test_schedules_supervise_their_contracted_exits(self):
        for schedule, supervised in (
            ("final_only", {2}),
            ("uniform", {1, 2}),
            ("progressive", {1, 2}),
        ):
            with self.subTest(schedule=schedule):
                config = replace(self.config, schedule=schedule)
                output = self.root / f"run-{schedule}"
                train(config, self.data, output)
                final = self.final_metrics(output)
                self.assertTrue(final["training_complete"])
                for exit_index in (1, 2):
                    key = f"val/exit_{exit_index}/move_loss"
                    if exit_index in supervised:
                        self.assertTrue(final[key] > 0, key)
                        self.assertTrue(final[f"val/exit_{exit_index}/value_loss"] > 0)
                    else:
                        self.assertNotIn(key, final)
                self.assertTrue(0.0 <= final["val/final_move_top1"] <= 1.0)

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

    def test_internals_and_diagnostics_survive(self):
        config = replace(self.config, internals=True)
        output = self.root / "instrumented"
        train(config, self.data, output)
        final = self.final_metrics(output)
        self.assertIn("internal/layer_2/move_relative_change", final)
        self.assertIn("internal/layer_00/residual_rms", final)
        self.assertIn("val/legal_top1", final)

    def test_conflict_probe_reports_pairwise_cosines(self):
        config = replace(self.config, schedule="uniform", conflict_probe=True)
        output = self.root / "conflict"
        train(config, self.data, output)
        final = self.final_metrics(output)
        # Two supervised exits yield exactly one pair; cosine lies in [-1, 1].
        self.assertTrue(-1.0 <= final["internal/grad_cosine_exit_1_2"] <= 1.0)

    def test_adapter_lr_scale_forms_a_separate_param_group(self):
        config = replace(
            self.config,
            model={**self.config.model, "exit_adapter": True},
            exit_adapter_lr_scale=0.1,
        )
        output = self.root / "lr-scale"
        checkpoint = read_checkpoint(train(config, self.data, output))
        groups = checkpoint["optimizer_state_dict"]["param_groups"]
        # Contract: adapters train at lr x scale, everything else at lr.
        self.assertEqual(len(groups), 2)
        self.assertAlmostEqual(groups[1]["initial_lr"] / groups[0]["initial_lr"], 0.1)

    def test_adapter_lr_scale_requires_adapters(self):
        with self.assertRaises(ValueError):
            replace(self.config, exit_adapter_lr_scale=0.5)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            replace(self.config, schedule="curriculum")
        with self.assertRaises(ValueError):
            replace(self.config, teacher_depth=0)
        # teacher_depth beyond the encoded depths is rejected by train().
        with self.assertRaises(ValueError):
            train(
                replace(self.config, teacher_depth=3), self.data, self.root / "deep"
            )
        # The depth trainer only accepts depth-architecture models.
        with self.assertRaises(ValueError):
            replace(
                self.config,
                model={"architecture": "base", "num_blocks": 1, "hidden_size": 16,
                       "intermediate_size": 32, "num_heads": 2},
            )


if __name__ == "__main__":
    unittest.main()
