import json
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch
from teacher_release import play, write_teacher_release

from pondera.chess.mapping import PIECE_TO_IDX, UCI_MOVE_TO_IDX
from pondera.chess.tokens import (
    PAD,
    SIDE,
    decode_board,
    encode_board,
    encode_boards,
)
from pondera.data.positions import TokenizedPositions
from pondera.data.teacher import prepare_teacher_policy
from pondera.inference.policy import pick_move
from pondera.models.board import black_to_move, reflect_board_tokens
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


def assert_modulo_fullmove(test, tokens, fen, repetition):
    """Decoded FEN matches every field except the fullmove counter."""
    decoded, decoded_repetition = decode_board(tokens)
    test.assertEqual(decoded.split()[:5], fen.split()[:5])
    test.assertEqual(decoded.split()[5], "1")
    test.assertEqual(decoded_repetition, min(repetition, 5))


class BoardModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_lossless_fields_and_counter_atoms(self):
        fens = [
            chess.STARTING_FEN,
            "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 99 257",
            "7k/Q7/8/8/8/8/8/K7 b - - 100 400",
        ]
        repetitions = [1, 4, 9]
        batch = encode_boards(fens, repetitions)
        for fen, rep, tokens in zip(fens, repetitions, batch):
            assert_modulo_fullmove(self, tokens, fen, rep)
        for bad in [
            "8/8/8/8/8/8/8/9 w - - 0 1",
            "8/8/8/8/8/8/8/8 x - - 0 1",
            "8/8/8/8/8/8/8/8 w - - -1 1",
            "8/8/8/8/8/8/8/8 w - - 101 1",
        ]:
            with self.assertRaises(ValueError):
                encode_board(bad)
        with self.assertRaises(ValueError):
            encode_board(chess.STARTING_FEN, 0)

    def test_sparse_atoms_match_python_chess_and_reject_duplicates(self):
        fen = chess.STARTING_FEN
        tokens = encode_board(fen, 1)
        self.assertEqual(len(tokens), 32 + 8)
        expected = sorted(
            square * 12 + PIECE_TO_IDX[piece.symbol()]
            for square, piece in chess.Board(fen).piece_map().items()
        )
        self.assertEqual(tokens[:32].tolist(), expected)
        self.assertEqual(tokens[32], SIDE)
        duplicated = np.concatenate((tokens[:2], tokens[:1], tokens[1:]))
        with self.assertRaises(ValueError):
            decode_board(duplicated)

    def test_orientation_is_reversible_and_matches_chess_mirror(self):
        fens = [
            "r3k2r/8/8/8/3pP3/8/8/R3K2R b Kq e3 99 400",
            "7k/8/8/8/8/8/p7/7K b - - 0 80",
            chess.STARTING_FEN,
        ]
        ids = torch.from_numpy(encode_boards(fens, [3, 2, 1]))
        black = black_to_move(ids)
        changed = reflect_board_tokens(ids, black)
        torch.testing.assert_close(reflect_board_tokens(changed, black), ids)
        for index, fen in enumerate(fens):
            expected = chess.Board(fen)
            if expected.turn == chess.BLACK:
                expected = expected.mirror()
            parts = expected.fen(en_passant="fen").split()
            # The side atom and the encoded counters keep the root frame.
            parts[1] = fen.split()[1]
            parts[5] = "1"
            assert_modulo_fullmove(
                self, changed[index], " ".join(parts), [3, 2, 1][index]
            )

    def test_embeddings_ignore_prefix_order_and_oriented_logits(self):
        fens = [
            chess.STARTING_FEN,
            "r3k2r/8/8/8/3pP3/8/8/R3K2R b Kq e3 99 400",
            "7k/8/8/8/8/8/p7/7K b - - 0 80",
        ]
        ids = torch.from_numpy(encode_boards(fens))
        config = dict(CONFIG)
        model = create_model(config).eval()
        logits, _ = model(ids=ids)
        order = torch.randperm(ids.shape[1])
        torch.testing.assert_close(logits, model(ids=ids[:, order])[0])
        reflected = model.reflected_actions
        torch.testing.assert_close(reflected[reflected], torch.arange(len(reflected)))
        for index, fen in enumerate(fens):
            board = chess.Board(fen)
            actual = {
                int(reflected[UCI_MOVE_TO_IDX[m.uci()]]) for m in board.legal_moves
            }
            expected = {UCI_MOVE_TO_IDX[m.uci()] for m in board.mirror().legal_moves}
            self.assertEqual(actual, expected)
        hidden, valid = model.embed(ids)
        output = model.backbone(hidden, valid=valid)
        canonical_logits = model.act_proj(output.hidden[:, -2])
        black = black_to_move(ids)
        expected = torch.where(
            black[:, None],
            canonical_logits[:, reflected],
            canonical_logits,
        )
        torch.testing.assert_close(logits, expected)
        logits.square().mean().backward()
        self.assertGreater(
            float(model.board_embedding.tokens.weight.grad.abs().sum()), 0
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pth"
            save_checkpoint(path, model, config)
            restored = load_model(path)
            torch.testing.assert_close(restored(ids=ids)[0], logits, rtol=0, atol=0)
            self.assertIn(
                chess.Move.from_uci(pick_move(restored, chess.Board())),
                chess.Board().legal_moves,
            )
            mate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
            self.assertEqual(pick_move(restored, mate), "0000")

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

    def test_teacher_encoded_training_budget_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            moves = [
                "e2e4",
                "e7e5",
                "g1f3",
                "b8c6",
                "f1b5",
                "a7a6",
                "b5a4",
                "g8f6",
            ]
            line = play(moves, repetitions=list(range(1, 9)))
            fens = [fen for fen, _, _ in line]
            fens[-1] = " ".join(fens[-1].split()[:4]) + " 99 400"
            train_records = [
                (fen, move, repetition)
                for fen, (_, move, repetition) in zip(fens, line)
            ]
            # Start the validation line after 1.d4 d5 2.c4 e6 to keep it disjoint.
            val_records = play(
                ["b1c3", "g8f6"],
                start="rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
            )
            data = root / "data"
            prepare_teacher_policy(
                write_teacher_release(
                    root, {"train": train_records, "val": val_records}, depth=1
                ),
                data,
                depth=1,
            )
            dataset = TokenizedPositions(data, "train")
            assert_modulo_fullmove(self, dataset.tokens[-1], fens[-1], 8)
            self.assertEqual(int(dataset.moves[-1]), UCI_MOVE_TO_IDX[moves[-1]])
            config = TrainConfig(
                model={**CONFIG, "dropout": 0.1},
                batch_size=2,
                accumulation_steps=2,
                epochs=3,
                workers=0,
                precision="fp32",
                val_every=2,
                log_every=1,
                save_every=1,
            )
            logical_lengths = [int((row != PAD).sum()) for row in dataset.tokens]
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
