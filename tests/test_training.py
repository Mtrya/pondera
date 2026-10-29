import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from teacher_release import play, write_teacher_release

from pondera.chess.tokens import CHESS_TOKENIZER_VERSION
from pondera.data.storage import load_manifest, write_manifest
from pondera.data.teacher import prepare_teacher_policy
from pondera.models.checkpoint import read_checkpoint
from pondera.training.supervised import TrainConfig, train

RUY_LOPEZ = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]
# Positions after 1.d4 d5 2.c4 e6, so the validation line is disjoint from the Ruy Lopez line.
EXCHANGE_START = "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"
EXCHANGE_VARIATION = ["b1c3", "g8f6", "c4d5", "e6d5", "g1f3", "b8c6", "f3e5", "c6e5"]


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = TrainConfig(
            model={
                "architecture": "base",
                "num_blocks": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "dropout": 0.1,
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
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data = self.root / "data"
        release = write_teacher_release(
            self.root,
            {
                "train": play(RUY_LOPEZ),
                "val": play(EXCHANGE_VARIATION, start=EXCHANGE_START),
            },
            depth=1,
        )
        prepare_teacher_policy(release, self.data, depth=1)

    def test_resume_matches_uninterrupted_training(self):
        expected = read_checkpoint(train(self.config, self.data, self.root / "full"))
        output = self.root / "resumed"
        interrupted = train(self.config, self.data, output, max_steps=1)
        halfway = read_checkpoint(
            train(self.config, self.data, output, resume=interrupted, max_steps=2)
        )
        self.assertEqual(halfway["step"], 2)
        unchanged = read_checkpoint(
            train(self.config, self.data, output, resume=interrupted, max_steps=2)
        )
        for key in ("step", "epoch", "next_micro", "scheduler_state_dict"):
            self.assertEqual(unchanged[key], halfway[key])
        for key, value in halfway["model_state_dict"].items():
            torch.testing.assert_close(
                unchanged["model_state_dict"][key], value, rtol=0, atol=0
            )
        actual = read_checkpoint(
            train(self.config, self.data, output, resume=interrupted)
        )
        self.assertEqual(actual["step"], 4)
        for key, value in expected["model_state_dict"].items():
            torch.testing.assert_close(
                actual["model_state_dict"][key], value, rtol=0, atol=0
            )
        self.assertEqual(
            actual["scheduler_state_dict"], expected["scheduler_state_dict"]
        )

    def test_board_diagnostics_and_legal_loss_resume(self):
        config = replace(
            self.config,
            model={**self.config.model, "architecture": "base", "num_blocks": 2},
            diagnostics=True,
            legal_loss=True,
            train_probe_positions=4,
        )
        expected = read_checkpoint(
            train(config, self.data, self.root / "diagnostic-full")
        )
        output = self.root / "diagnostic-resumed"
        interrupted = train(config, self.data, output, max_steps=1)
        partial = json.loads((output / "metrics.jsonl").read_text().splitlines()[-1])
        self.assertFalse(partial["training_complete"])
        actual = read_checkpoint(train(config, self.data, output, resume=interrupted))
        for key, value in expected["model_state_dict"].items():
            torch.testing.assert_close(
                actual["model_state_dict"][key], value, rtol=0, atol=0
            )
        final = json.loads((output / "metrics.jsonl").read_text().splitlines()[-1])
        self.assertTrue(final["training_complete"])
        self.assertEqual(final["val/positions"], 8)
        self.assertAlmostEqual(
            final["val/loss"],
            final["val/legal_loss"] + final["val/legality_penalty"],
            places=5,
        )
        self.assertIn("internal/layer_2/move_relative_change", final)
        self.assertIn("generalization/legal_loss_gap", final)
        self.assertTrue((output / "probes" / "val-000004.npz").exists())

    def test_training_rejects_unversioned_or_incompatible_data(self):
        path = self.data / "manifest.json"
        manifest = load_manifest(path)
        for version in (None, CHESS_TOKENIZER_VERSION + 1):
            with self.subTest(version=version):
                invalid = dict(manifest)
                if version is None:
                    del invalid["tokenizer_version"]
                else:
                    invalid["tokenizer_version"] = version
                write_manifest(path, invalid)
                with self.assertRaises(ValueError):
                    train(self.config, self.data, self.root / "rejected")
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            train(self.config, self.data, self.root / "rejected")

    def test_resume_rejects_incompatible_checkpoint(self):
        output = self.root / "resumed"
        path = train(self.config, self.data, output, max_steps=1)
        checkpoint = read_checkpoint(path)
        for corruption in (
            {"format_version": None},
            {"format_version": 2},
            {"tokenizer_version": 1},
            {"model_config": None},
        ):
            with self.subTest(corruption=corruption):
                invalid = dict(checkpoint)
                for key, value in corruption.items():
                    if value is None:
                        del invalid[key]
                    else:
                        invalid[key] = value
                torch.save(invalid, path)
                with self.assertRaises(ValueError):
                    train(self.config, self.data, output, resume=path)

    def test_nonpositive_step_limit_is_rejected(self):
        for max_steps in (0, -1):
            with self.subTest(max_steps=max_steps):
                with self.assertRaises(ValueError):
                    train(
                        self.config,
                        self.data,
                        self.root / "rejected",
                        max_steps=max_steps,
                    )

    def test_internals_probe_writes_scalars_and_arrays(self):
        config = replace(self.config, internals=True)
        output = self.root / "internals"
        train(config, self.data, output)
        rows = [
            json.loads(line)
            for line in (output / "metrics.jsonl").read_text().splitlines()
        ]
        final = rows[-1]
        self.assertIn("internal/layer_00/residual_rms", final)
        self.assertIn("internal/layer_00/attention_entropy_mean", final)
        self.assertIn("internal/layer_00/ffn_dead_fraction", final)
        self.assertIn("internal/embedding/tokens_starved_fraction", final)
        self.assertIn("internal/drift/embedding", final)
        probes = output / "probes"
        self.assertTrue((probes / "init-embeddings.npz").exists())
        artifacts = sorted(probes.glob("internal-*.npz"))
        self.assertTrue(artifacts)
        arrays = np.load(artifacts[-1])
        self.assertIn("attention_entropy", arrays)
        self.assertIn("rank_spectrum_00", arrays)

    def test_internals_probe_survives_resume(self):
        config = replace(self.config, internals=True)
        output = self.root / "internals-resumed"
        interrupted = train(config, self.data, output, max_steps=1)
        train(config, self.data, output, resume=interrupted)
        final = json.loads((output / "metrics.jsonl").read_text().splitlines()[-1])
        self.assertTrue(final["training_complete"])
        self.assertIn("internal/layer_00/residual_rms", final)

    def test_resume_can_regroup_microbatches_without_changing_updates(self):
        small = replace(self.config, model={**self.config.model, "dropout": 0.0})
        large = replace(small, batch_size=4, accumulation_steps=1)
        expected = read_checkpoint(train(large, self.data, self.root / "large"))
        output = self.root / "regrouped"
        interrupted = train(small, self.data, output, max_steps=1)
        actual = read_checkpoint(train(large, self.data, output, resume=interrupted))
        self.assertEqual(actual["step"], expected["step"])
        self.assertEqual(actual["examples_seen"], expected["examples_seen"])
        self.assertEqual(actual["next_micro"], expected["next_micro"])
        self.assertEqual(
            actual["execution_history"],
            [
                {"step": 0, "batch_size": 2, "accumulation_steps": 2},
                {"step": 1, "batch_size": 4, "accumulation_steps": 1},
            ],
        )
        for key, value in expected["model_state_dict"].items():
            torch.testing.assert_close(
                actual["model_state_dict"][key], value, atol=3e-6, rtol=3e-5
            )
        with self.assertRaises(ValueError):
            train(replace(large, batch_size=2), self.data, output, resume=interrupted)


class TrainConfigTests(unittest.TestCase):
    def test_invalid_optimizer_parameters_are_rejected(self):
        invalid = {
            "learning_rate": (0, -1, float("nan"), float("inf")),
            "grad_clip": (0, -1, float("nan"), float("inf")),
            "weight_decay": (-1, float("nan"), float("inf")),
            "warmup_ratio": (-0.1, 1.1, float("nan"), float("inf")),
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        TrainConfig(**{name: value})
        for warmup_ratio in (0, 1):
            TrainConfig(weight_decay=0, warmup_ratio=warmup_ratio)
