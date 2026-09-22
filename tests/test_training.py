import tempfile
import unittest
from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from pondera.chess.tokenize import TOKENIZER_VERSION
from pondera.data.prepare import prepare_dataset
from pondera.data.storage import load_manifest, write_manifest
from pondera.models.checkpoint import read_checkpoint
from pondera.training.supervised import TrainConfig, train


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = TrainConfig(
            model={
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
        board = chess.Board()
        fens = []
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]
        for move in moves:
            fens.append(board.fen())
            board.push_uci(move)
        source = self.root / "positions.parquet"
        pq.write_table(pa.table({"fen": fens, "next_move": moves}), source)
        prepare_dataset([source], [source], self.data)

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

    def test_training_rejects_unversioned_or_incompatible_data(self):
        path = self.data / "manifest.json"
        manifest = load_manifest(path)
        for version in (None, TOKENIZER_VERSION + 1):
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
        checkpoint["tokenizer_version"] = TOKENIZER_VERSION + 1
        torch.save(checkpoint, path)
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
