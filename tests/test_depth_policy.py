import tempfile
import unittest
from pathlib import Path

import chess
import torch

from pondera.chess.tokens import encode_boards
from pondera.chess.mapping import UCI_MOVE_TO_IDX
from pondera.models.checkpoint import load_model, save_checkpoint
from pondera.models.factory import create_model
from pondera.training.cost import depth_flops

CONFIG = {
    "architecture": "depth",
    "cores": [{"layers": 2, "repetition": {"max_repetitions": 3}}],
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_heads": 2,
}


class DepthPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = create_model(CONFIG).eval()
        self.ids = torch.from_numpy(
            encode_boards(
                [
                    chess.STARTING_FEN,
                    "r3k2r/8/8/3pP3/8/8/8/R3K2R b Kq e3 130 400",
                ]
            )
        )

    def test_exits_match_repetitions_and_vocabulary_slices(self):
        exits, _ = self.model.forward_exits(ids=self.ids)
        self.assertEqual(len(exits), 3)
        for exit_ in exits:
            self.assertEqual(exit_.move_logits.shape, (2, len(UCI_MOVE_TO_IDX)))
            self.assertEqual(exit_.value_logits.shape, (2, 80))
            self.assertTrue(bool(exit_.active.all()))

    def test_final_exit_matches_the_fast_path(self):
        exits, _ = self.model.forward_exits(ids=self.ids)
        move, value = self.model(ids=self.ids)
        torch.testing.assert_close(exits[-1].move_logits, move)
        torch.testing.assert_close(exits[-1].value_logits, value)

    def test_reflection_table_matches_python_chess_mirrors(self):
        # Independent source: python-chess legal moves of mirrored boards.
        board = chess.Board("r3k2r/8/8/3pP3/8/8/8/R3K2R b Kq e3 130 400")
        reflected = self.model.reflected_actions
        actual = {int(reflected[UCI_MOVE_TO_IDX[m.uci()]]) for m in board.legal_moves}
        expected = {UCI_MOVE_TO_IDX[m.uci()] for m in board.mirror().legal_moves}
        self.assertEqual(actual, expected)

    def test_serialization_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.pth"
            save_checkpoint(path, self.model, CONFIG)
            torch.testing.assert_close(
                load_model(path)(ids=self.ids)[0], self.model(ids=self.ids)[0]
            )
            self.model.save_pretrained(Path(directory) / "hub")
            torch.testing.assert_close(
                load_model(Path(directory) / "hub")(ids=self.ids)[0],
                self.model(ids=self.ids)[0],
            )


class DepthFlopsTests(unittest.TestCase):
    def test_hand_derived_literal(self):
        # Effective layers: 2 layers x 3 repetitions = 6. Length 10 + 2 readouts = 12.
        # Per layer: 8*12*16^2 + 6*12*16*32 + 4*12^2*16 = 24576+36864+9216 = 70656.
        # Trunk: 6*70656 = 423936. Head: 3 exits x 2 positions x 2*16*2398 = 460416.
        # Training: 3*(423936+460416) = 2653056; evaluation: 884352.
        config = {
            "architecture": "depth",
            "hidden_size": 16,
            "intermediate_size": 32,
            "cores": [{"layers": 2, "repetition": {"max_repetitions": 3}}],
        }
        self.assertEqual(
            float(depth_flops(config, [10], supervised_exits=3)[0]), 2653056.0
        )
        self.assertEqual(
            float(
                depth_flops(config, [10], supervised_exits=3, training=False)[0]
            ),
            884352.0,
        )


if __name__ == "__main__":
    unittest.main()
