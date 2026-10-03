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


ADAPTER_CONFIG = dict(CONFIG, exit_adapter=True)


class DepthExitAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = create_model(ADAPTER_CONFIG).eval()
        self.ids = torch.from_numpy(
            encode_boards(
                [
                    chess.STARTING_FEN,
                    "r3k2r/8/8/3pP3/8/8/8/R3K2R b Kq e3 130 400",
                ]
            )
        )

    def test_one_adapter_per_core_exit(self):
        # Contract: exits follow core repetition boundaries, one adapter each.
        self.assertEqual(len(self.model.exit_adapters), 3)

    def test_final_exit_matches_the_fast_path(self):
        exits, _ = self.model.forward_exits(ids=self.ids)
        move, value = self.model(ids=self.ids)
        torch.testing.assert_close(exits[-1].move_logits, move)
        torch.testing.assert_close(exits[-1].value_logits, value)

    def test_serialization_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pth"
            save_checkpoint(path, self.model, ADAPTER_CONFIG)
            torch.testing.assert_close(
                load_model(path)(ids=self.ids)[0], self.model(ids=self.ids)[0]
            )

    def test_adapters_default_off(self):
        self.assertIsNone(create_model(CONFIG).exit_adapters)


class DepthAdapterVariantTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.ids = torch.from_numpy(encode_boards([chess.STARTING_FEN]))

    def test_zero_init_residual_is_exact_identity(self):
        # Invariant: zero-initialized residual adapters pass the state through
        # untouched, so at init the model matches the adapter-free model built
        # under the same seed (adapters are constructed last).
        torch.manual_seed(640)
        plain = create_model(CONFIG).eval()
        torch.manual_seed(640)
        adapted = create_model(
            dict(CONFIG, exit_adapter=True, exit_adapter_init="zero")
        ).eval()
        torch.testing.assert_close(adapted(ids=self.ids)[0], plain(ids=self.ids)[0])

    def test_gated_adapter_is_exact_identity_at_init(self):
        torch.manual_seed(640)
        plain = create_model(CONFIG).eval()
        torch.manual_seed(640)
        gated = create_model(
            dict(CONFIG, exit_adapter=True, exit_adapter_mode="gated")
        ).eval()
        torch.testing.assert_close(gated(ids=self.ids)[0], plain(ids=self.ids)[0])
        self.assertEqual(len(gated.exit_adapter_gates), 3)

    def test_replace_mode_roundtrip(self):
        config = dict(CONFIG, exit_adapter=True, exit_adapter_mode="replace")
        model = create_model(config).eval()
        exits, _ = model.forward_exits(ids=self.ids)
        move, value = model(ids=self.ids)
        torch.testing.assert_close(exits[-1].move_logits, move)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replace.pth"
            save_checkpoint(path, model, config)
            torch.testing.assert_close(
                load_model(path)(ids=self.ids)[0], model(ids=self.ids)[0]
            )

    def test_invalid_adapter_combinations_rejected(self):
        with self.assertRaises(ValueError):
            create_model(dict(CONFIG, exit_adapter_mode="gated"))
        with self.assertRaises(ValueError):
            create_model(
                dict(
                    CONFIG,
                    exit_adapter=True,
                    exit_adapter_mode="replace",
                    exit_adapter_init="zero",
                )
            )
        with self.assertRaises(ValueError):
            create_model(
                dict(CONFIG, exit_adapter=True, exit_adapter_mode="prepend")
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
        # Trunk: 6*70656 = 423936. Head: 3 exits x 2 positions x 2*16*2400 = 460800.
        # Training: 3*(423936+460800) = 2654208; evaluation: 884736.
        config = {
            "architecture": "depth",
            "hidden_size": 16,
            "intermediate_size": 32,
            "cores": [{"layers": 2, "repetition": {"max_repetitions": 3}}],
        }
        self.assertEqual(
            float(depth_flops(config, [10], head_exits=3)[0]), 2654208.0
        )
        self.assertEqual(
            float(
                depth_flops(config, [10], head_exits=3, training=False)[0]
            ),
            884736.0,
        )

    def test_repetitions_override_scales_trunk(self):
        # Same per-layer cost 70656 as above. Override to 2 executed repetitions:
        # trunk 2*2*70656 = 282624; head 2 exits x 2 positions x 2*16*2400 = 307200.
        # Training: 3*(282624+307200) = 1769472. Override equal to the configured
        # maximum must reproduce the un-overridden value exactly.
        config = {
            "architecture": "depth",
            "hidden_size": 16,
            "intermediate_size": 32,
            "cores": [{"layers": 2, "repetition": {"max_repetitions": 3}}],
        }
        self.assertEqual(
            float(depth_flops(config, [10], head_exits=2, repetitions=2)[0]),
            1769472.0,
        )
        self.assertEqual(
            float(depth_flops(config, [10], head_exits=3, repetitions=3)[0]),
            float(depth_flops(config, [10], head_exits=3)[0]),
        )


class RepetitionLimitTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = create_model(CONFIG).eval()
        self.ids = torch.from_numpy(encode_boards([chess.STARTING_FEN]))

    def test_truncated_exits_match_full_run_prefix(self):
        # Invariant: stopping after k repetitions must reproduce the first k
        # exits of the untruncated run — the truncation path and the full path
        # are two sources for the same computation.
        full, _ = self.model.forward_exits(ids=self.ids)
        self.assertEqual(len(full), 3)
        for k in (1, 2, 3):
            truncated, _ = self.model.forward_exits(
                ids=self.ids, repetition_limit=k
            )
            self.assertEqual(len(truncated), k)
            for index in range(k):
                torch.testing.assert_close(
                    truncated[index].move_logits, full[index].move_logits
                )
                torch.testing.assert_close(
                    truncated[index].value_logits, full[index].value_logits
                )

    def test_truncated_exits_match_full_run_prefix_under_autocast(self):
        # The sparse KV path mixes autocast dtypes; the prefix invariant must
        # hold there too (this is the training configuration).
        with torch.autocast("cpu", torch.bfloat16):
            full, _ = self.model.forward_exits(ids=self.ids)
            for k in (1, 2, 3):
                truncated, _ = self.model.forward_exits(
                    ids=self.ids, repetition_limit=k
                )
                self.assertEqual(len(truncated), k)
                for index in range(k):
                    torch.testing.assert_close(
                        truncated[index].move_logits, full[index].move_logits
                    )

    def test_scalar_and_per_token_limits_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            self.model.forward_exits(
                ids=self.ids, repetition_limit=1, repetition_limits=[None]
            )


if __name__ == "__main__":
    unittest.main()
