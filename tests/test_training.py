import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch

from pondera.chess.mapping import UCI_MOVE_TO_IDX
from pondera.chess.tokenize import encode_fens
from pondera.models.checkpoint import read_checkpoint
from pondera.training.supervised import TrainConfig, train


class TrainingTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_training(self):
        torch.set_num_threads(1)
        config = TrainConfig(
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
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            board = chess.Board()
            fens, moves = [], []
            for move in [
                "e2e4",
                "e7e5",
                "g1f3",
                "b8c6",
                "f1b5",
                "a7a6",
                "b5a4",
                "g8f6",
            ]:
                fens.append(board.fen())
                moves.append(UCI_MOVE_TO_IDX[move])
                board.push_uci(move)
            for split in ("train", "val"):
                np.save(data / f"tokens_{split}.npy", encode_fens(fens))
                np.save(data / f"moves_{split}.npy", np.array(moves, dtype=np.int16))
            expected = read_checkpoint(train(config, data, root / "full"))
            interrupted = train(config, data, root / "resumed", max_steps=1)
            actual = read_checkpoint(
                train(config, data, root / "resumed", resume=interrupted)
            )
            self.assertEqual(actual["step"], 4)
            for key, value in expected["model_state_dict"].items():
                torch.testing.assert_close(
                    actual["model_state_dict"][key], value, rtol=0, atol=0
                )
            self.assertEqual(
                actual["scheduler_state_dict"], expected["scheduler_state_dict"]
            )
