import tempfile
import unittest
from pathlib import Path

import chess
import torch

from pondera.chess.codec import encode_boards
from pondera.chess.domain import SQUARE_TO_IDX
from pondera.chess.vocabulary import EN_PASSANT, HALFMOVE, UCI_MOVE_TO_IDX
from pondera.inference.policy import pick_move
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
                    "r3k2r/8/8/3pP3/8/8/8/R3K2R b Kq e3 99 400",
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

    def test_uci_move_selection_rejects_sequence_models(self):
        sequence = create_model(
            {
                "architecture": "sequence",
                "num_blocks": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
            }
        )
        with self.assertRaises(ValueError):
            pick_move(sequence, chess.Board())

    def test_move_selection_encodes_en_passant_without_a_legal_capture(self):
        board = chess.Board("r3k2r/8/8/3pP3/8/8/8/R3K2R b Kq e3 99 400")
        # python-chess omits the ep square from its default FEN when no legal
        # ep capture exists; the policy input must keep the stored field.
        self.assertEqual(board.fen().split()[3], "-")
        captured = []

        def capture(module, args, kwargs):
            captured.append(kwargs["ids"])

        handle = self.model.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            pick_move(self.model, board)
        finally:
            handle.remove()
        tokens = captured[0][0].tolist()
        ep_atoms = [token for token in tokens if EN_PASSANT <= token < HALFMOVE]
        self.assertEqual(ep_atoms, [EN_PASSANT + SQUARE_TO_IDX["e3"]])

    def test_reflection_table_matches_python_chess_mirrors(self):
        # Independent source: python-chess legal moves of mirrored boards.
        board = chess.Board("r3k2r/8/8/3pP3/8/8/8/R3K2R b Kq e3 99 400")
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


AUX_CONFIG = dict(CONFIG, board_aux=True)


class DepthBoardAuxTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = create_model(AUX_CONFIG).eval()
        self.ids = torch.from_numpy(encode_boards([chess.STARTING_FEN]))

    def test_board_aux_shapes_and_fast_path_invariant(self):
        # Contract: 13 square-content classes per board slot; move/value unchanged.
        exits, _ = self.model.forward_exits(ids=self.ids)
        board_width = exits[-1].board_logits.shape[1]
        for exit_ in exits:
            self.assertEqual(exit_.board_logits.shape[-1], 13)
            self.assertEqual(exit_.active_board.shape, (1, board_width - 2))
        move, value = self.model(ids=self.ids)
        torch.testing.assert_close(exits[-1].move_logits, move)
        torch.testing.assert_close(exits[-1].value_logits, value)

    def test_board_aux_serialization_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aux.pth"
            save_checkpoint(path, self.model, AUX_CONFIG)
            loaded = load_model(path)
            exits, _ = loaded.forward_exits(ids=self.ids)
            expected, _ = self.model.forward_exits(ids=self.ids)
            torch.testing.assert_close(
                exits[-1].board_logits, expected[-1].board_logits
            )

    def test_board_aux_default_off(self):
        exits, _ = create_model(CONFIG).forward_exits(ids=self.ids)
        self.assertIsNone(exits[-1].board_logits)


class DepthFlopsTests(unittest.TestCase):
    def test_hand_derived_literal(self):
        # Effective layers: 2 layers x 3 repetitions = 6. Length 10 + 2 readouts = 12.
        # Per layer: 8*12*16^2 + 6*12*16*32 + 4*12^2*16 = 24576+36864+9216 = 70656.
        # Trunk: 6*70656 = 423936. Head: 3 exits x 2 positions x 2*16*3006 = 577152.
        # Training: 3*(423936+577152) = 3003264; evaluation: 1001088.
        config = {
            "architecture": "depth",
            "hidden_size": 16,
            "intermediate_size": 32,
            "cores": [{"layers": 2, "repetition": {"max_repetitions": 3}}],
        }
        self.assertEqual(float(depth_flops(config, [10], head_exits=3)[0]), 3003264.0)
        self.assertEqual(
            float(depth_flops(config, [10], head_exits=3, training=False)[0]),
            1001088.0,
        )

    def test_repetitions_override_scales_trunk(self):
        # Same per-layer cost 70656 as above. Override to 2 executed repetitions:
        # trunk 4*70656 = 282624; head 2 exits x 2 positions x 2*16*3006 = 384768.
        # Training: 3*(282624+384768) = 2002176. Override equal to the configured
        # maximum must reproduce the un-overridden value exactly.
        config = {
            "architecture": "depth",
            "hidden_size": 16,
            "intermediate_size": 32,
            "cores": [{"layers": 2, "repetition": {"max_repetitions": 3}}],
        }
        self.assertEqual(
            float(depth_flops(config, [10], head_exits=2, repetitions=2)[0]),
            2002176.0,
        )
        self.assertEqual(
            float(depth_flops(config, [10], head_exits=3, repetitions=3)[0]),
            float(depth_flops(config, [10], head_exits=3)[0]),
        )


if __name__ == "__main__":
    unittest.main()
