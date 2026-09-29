import json
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch
from teacher_release import play, write_teacher_release

from pondera.chess.board_tokens import PAD, decode_board, encode_board, encode_boards
from pondera.chess.mapping import EMPTY_SQ_IDX, UCI_MOVE_TO_IDX
from pondera.data.positions import TokenizedPositions
from pondera.data.teacher import prepare_teacher_policy
from pondera.inference.policy import pick_move
from pondera.models.board import black_to_move, pack_board_tokens, reflect_board_tokens
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

    def test_orientation_is_reversible_and_matches_chess_mirror(self):
        fens = [
            "r3k2r/8/8/8/3pP3/8/8/R3K2R b Kq e3 130 400",
            "7k/8/8/8/8/8/p7/7K b - - 0 80",
            chess.STARTING_FEN,
        ]
        ids = torch.from_numpy(encode_boards(fens, [3, 2, 1]))
        for layout in ("squares", "pieces"):
            packed = pack_board_tokens(ids, layout)
            black = black_to_move(**packed)
            changed, positions = reflect_board_tokens(**packed, reflected=black)
            recovered, recovered_positions = reflect_board_tokens(
                changed, positions, black
            )
            torch.testing.assert_close(recovered, packed["ids"])
            torch.testing.assert_close(recovered_positions, packed["positions"])
            for index, fen in enumerate(fens):
                restored = torch.full_like(ids[index], PAD)
                restored[:64] = EMPTY_SQ_IDX
                valid = changed[index] != PAD
                restored[positions[index, valid]] = changed[index, valid]
                transformed, repetition = decode_board(restored)
                expected = chess.Board(fen)
                if expected.turn == chess.BLACK:
                    expected = expected.mirror()
                parts = expected.fen(en_passant="fen").split()
                parts[1] = fen.split()[1]
                self.assertEqual(transformed, " ".join(parts))
                self.assertEqual(repetition, [3, 2, 1][index])

    def test_joint_embeddings_and_oriented_policy_roundtrip(self):
        fens = [
            chess.STARTING_FEN,
            "r3k2r/8/8/8/3pP3/8/8/R3K2R b Kq e3 130 400",
            "7k/8/8/8/8/8/p7/7K b - - 0 80",
        ]
        ids = torch.from_numpy(encode_boards(fens))
        for encoding in ("additive", "joint"):
            config = {
                **CONFIG,
                "encoding": encoding,
                "orientation": "side_to_move",
                "layout": "pieces",
            }
            model = create_model(config).eval()
            packed = pack_board_tokens(ids, "pieces")
            logits, _ = model(**packed)
            torch.testing.assert_close(logits, model(ids=ids)[0])
            order = torch.arange(packed["ids"].shape[1] - 1, -1, -1)
            torch.testing.assert_close(
                logits, model(**{k: v[:, order] for k, v in packed.items()})[0]
            )
            reflected = model.reflected_actions
            torch.testing.assert_close(
                reflected[reflected], torch.arange(len(reflected))
            )
            for index, fen in enumerate(fens):
                board = chess.Board(fen)
                actual = {
                    int(reflected[UCI_MOVE_TO_IDX[m.uci()]]) for m in board.legal_moves
                }
                expected = {
                    UCI_MOVE_TO_IDX[m.uci()] for m in board.mirror().legal_moves
                }
                self.assertEqual(actual, expected)
            hidden, valid, squares = model.embed(**packed)
            output = model.backbone(hidden, valid=valid, squares=squares)
            canonical_logits = model.act_proj(output.hidden[:, -2])
            torch.testing.assert_close(logits[0], canonical_logits[0])
            torch.testing.assert_close(logits[1:], canonical_logits[1:, reflected])
            logits.square().mean().backward()
            self.assertGreater(
                float(model.board_embedding.squares.weight.grad.abs().sum()), 0
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "policy.pth"
                save_checkpoint(path, model, config)
                torch.testing.assert_close(
                    load_model(path)(ids=ids)[0], logits, rtol=0, atol=0
                )

    def test_planar_geometry_variants(self):
        from pondera.models.blocks import Rope2D

        # Rotary invariant: the rotated dot product depends only on the
        # per-axis displacement, not on the absolute squares.
        rope = Rope2D(8)
        torch.manual_seed(0)
        q = torch.randn(1, 1, 1, 8)
        k = torch.randn(1, 1, 1, 8)

        def score(query_square, key_square):
            squares = torch.tensor([[query_square]])
            rq = rope(q, squares)
            rk = rope(k, torch.tensor([[key_square]]))
            return float(rq @ rk.transpose(-1, -2))

        # a1=0 -> file = index % 8, rank = index // 8 (same convention as the encoder)
        def square(rank, file):
            return rank * 8 + file

        pairs = [
            ((square(0, 0), square(1, 1)), (square(2, 2), square(3, 3))),  # (+1, +1)
            ((square(1, 0), square(3, 2)), (square(3, 1), square(5, 3))),  # (+2, +2)
            ((square(7, 7), square(4, 4)), (square(7, 6), square(4, 3))),  # (-3, -3)
        ]
        for (q1, k1), (q2, k2) in pairs:
            self.assertAlmostEqual(score(q1, k1), score(q2, k2), places=5)
        # Non-board tokens (square -1) must get the identity rotation.
        arbitrary = torch.randn(2, 3, 5, 8)
        offboard = torch.full((2, 5), -1, dtype=torch.long)
        torch.testing.assert_close(rope(arbitrary, offboard), arbitrary)

        ids = torch.from_numpy(encode_boards([chess.STARTING_FEN]))
        baseline = create_model(CONFIG)
        for geometry in ("sin2d", "rope2d"):
            model = create_model({**CONFIG, "geometry": geometry}).eval()
            self.assertFalse(hasattr(model.board_embedding, "squares"))
            logits, _ = model(ids=ids)
            self.assertEqual(logits.shape[1], len(UCI_MOVE_TO_IDX))
            self.assertTrue(torch.isfinite(logits).all())
            logits.square().mean().backward()
            self.assertGreater(
                float(model.board_embedding.tokens.weight.grad.abs().sum()), 0
            )
        self.assertTrue(hasattr(baseline.board_embedding, "squares"))
        # The fixed 2D grid is a frozen contract: identical across instances.
        first = create_model({**CONFIG, "geometry": "sin2d"})
        second = create_model({**CONFIG, "geometry": "sin2d"})
        torch.testing.assert_close(
            first.board_embedding.squares_fixed, second.board_embedding.squares_fixed
        )
        with self.assertRaises(ValueError):
            create_model({**CONFIG, "geometry": "rope2d", "encoding": "joint"})

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
                    mate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
                    self.assertEqual(pick_move(restored, mate), "0000")
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
            fens[-1] = " ".join(fens[-1].split()[:4]) + " 130 400"
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
