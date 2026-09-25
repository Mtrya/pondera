import json
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from pondera.chess.board_tokens import PAD, decode_board, encode_board, encode_boards
from pondera.chess.mapping import EMPTY_SQ_IDX, UCI_MOVE_TO_IDX
from pondera.data.positions import TokenizedPositions
from pondera.data.prepare import prepare_dataset
from pondera.inference.policy import pick_move
from pondera.models.board import pack_board_tokens
from pondera.models.checkpoint import load_model, read_checkpoint, save_checkpoint
from pondera.models.factory import create_model
from pondera.training.cost import policy_flops
from pondera.training.supervised import TrainConfig, train

CONFIG = {
    "architecture": "base",
    "num_blocks": 2,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_heads": 2,
}


class BoardModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_lossless_counters_special_fields_and_promotions(self):
        fens = [
            chess.STARTING_FEN,
            "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 128 257",
            f"7k/Q7/8/8/8/8/8/K7 b - - {2**100} {2**150 + 13}",
        ]
        repetitions = [1, 7, 2**91 + 5]
        batch = encode_boards(fens, repetitions)
        for fen, rep, tokens in zip(fens, repetitions, batch):
            self.assertEqual(decode_board(tokens), (fen, rep))
        for bad in [
            "8/8/8/8/8/8/8/9 w - - 0 1",
            "8/8/8/8/8/8/8/8 x - - 0 1",
            "8/8/8/8/8/8/8/8 w - - -1 1",
        ]:
            with self.assertRaises(ValueError):
                encode_board(bad)

    def test_piece_packing_preserves_coordinates_and_metadata(self):
        fens = [chess.STARTING_FEN, "7k/Q7/8/8/8/8/8/K7 b - - 129 999"]
        ids = torch.from_numpy(encode_boards(fens))
        packed = pack_board_tokens(ids, "pieces")
        self.assertLess(packed["ids"].shape[1], ids.shape[1])
        for i, fen in enumerate(fens):
            restored = torch.full_like(ids[i], PAD)
            restored[:64] = EMPTY_SQ_IDX
            valid = packed["ids"][i] != PAD
            restored[packed["positions"][i, valid]] = packed["ids"][i, valid]
            self.assertEqual(decode_board(restored), (fen, 1))

    def test_policy_gradients_padding_and_checkpoint_roundtrip(self):
        fens = [chess.STARTING_FEN, "7k/Q7/8/8/8/8/8/K7 b - - 129 999"]
        ids = torch.from_numpy(encode_boards(fens))
        for layout, geometry in [
            ("squares", "absolute"),
            ("squares", "relative"),
            ("pieces", "relative"),
        ]:
            with self.subTest(layout=layout, geometry=geometry):
                config = {**CONFIG, "layout": layout, "geometry": geometry}
                model = create_model(config).eval()
                expected = model(ids=ids)
                packed = pack_board_tokens(ids, layout)
                padded = torch.cat(
                    (ids, torch.full((2, 4), PAD, dtype=ids.dtype)), dim=1
                )
                torch.testing.assert_close(model(**packed)[0], expected[0])
                torch.testing.assert_close(model(ids=padded)[0], expected[0])
                loss = torch.nn.functional.cross_entropy(
                    expected[0], torch.tensor([0, 1])
                )
                loss.backward()
                self.assertGreater(
                    float(model.board_embedding.tokens.weight.grad.abs().sum()), 0
                )
                if geometry == "relative":
                    self.assertGreater(
                        float(
                            model.backbone.cores[0]
                            .layers[0]
                            .geometry.embedding.weight.grad.abs()
                            .sum()
                        ),
                        0,
                    )
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "model.pth"
                    save_checkpoint(path, model, config)
                    restored = load_model(path)
                    torch.testing.assert_close(
                        restored(ids=ids)[0], expected[0], rtol=0, atol=0
                    )
                    self.assertIn(
                        chess.Move.from_uci(pick_move(restored, chess.Board())),
                        chess.Board().legal_moves,
                    )
                    hub = Path(directory) / "hub"
                    model.save_pretrained(hub)
                    torch.testing.assert_close(
                        load_model(hub)(ids=ids)[0], expected[0], rtol=0, atol=0
                    )

    def test_depth_policy_serialization(self):
        config = {key: value for key, value in CONFIG.items() if key != "num_blocks"}
        config.update(
            architecture="depth",
            cores=[{"layers": 2, "repetition": {"max_repetitions": 3}}],
        )
        model = create_model(config).eval()
        ids = torch.from_numpy(encode_boards([chess.STARTING_FEN]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.pth"
            save_checkpoint(path, model, config)
            torch.testing.assert_close(load_model(path)(ids=ids)[0], model(ids=ids)[0])
            model.save_pretrained(Path(directory) / "hub")
            torch.testing.assert_close(
                load_model(Path(directory) / "hub")(ids=ids)[0], model(ids=ids)[0]
            )

    def test_real_parquet_lossless_training_budget_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fens, moves = [], []
            board = chess.Board()
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
                moves.append(move)
                board.push_uci(move)
            fens[-1] = " ".join(fens[-1].split()[:4]) + " 130 400"
            source = root / "source.parquet"
            pq.write_table(
                pa.table(
                    {"fen": fens, "next_move": moves, "repetition": list(range(1, 9))}
                ),
                source,
            )
            data = root / "data"
            prepare_dataset([source], [source], data, tokenizer_version=2)
            dataset = TokenizedPositions(data, "train", 2)
            self.assertEqual(decode_board(dataset.tokens[-1]), (fens[-1], 8))
            self.assertEqual(int(dataset.moves[-1]), UCI_MOVE_TO_IDX[moves[-1]])
            config = TrainConfig(
                model={**CONFIG, "layout": "pieces", "dropout": 0.1},
                batch_size=2,
                accumulation_steps=2,
                epochs=3,
                workers=0,
                precision="fp32",
                val_every=2,
                log_every=1,
                save_every=1,
            )
            logical_lengths = [
                int((row != PAD).sum()) - int((row[:64] == EMPTY_SQ_IDX).sum())
                for row in dataset.tokens
            ]
            config.train_flops_budget = (
                float(policy_flops(config.model, logical_lengths).sum()) * 1.4
            )
            complete = read_checkpoint(train(config, data, root / "full"))
            interrupted = train(config, data, root / "resume", max_steps=1)
            resumed = read_checkpoint(
                train(config, data, root / "resume", resume=interrupted)
            )
            self.assertGreaterEqual(resumed["train_flops"], config.train_flops_budget)
            self.assertEqual(resumed["train_flops"], complete["train_flops"])
            self.assertEqual(resumed["step"], 3)
            for key, value in complete["model_state_dict"].items():
                torch.testing.assert_close(
                    resumed["model_state_dict"][key], value, rtol=0, atol=0
                )
            metrics = [
                json.loads(line)
                for line in (root / "full/metrics.jsonl").read_text().splitlines()
            ]
            self.assertTrue(metrics[-1]["final"])
            self.assertTrue(np.isfinite(metrics[-1]["val/loss"]))


if __name__ == "__main__":
    unittest.main()
