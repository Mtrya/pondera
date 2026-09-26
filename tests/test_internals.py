import math
import unittest

import torch

from pondera.chess.board_tokens import encode_boards
from pondera.models.board import pack_board_tokens
from pondera.models.policy import BoardPolicyModel
from pondera.training.internals import InternalProbe, effective_rank

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def tiny_model():
    torch.manual_seed(640)
    return BoardPolicyModel(
        num_blocks=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        dropout=0.0,
    )


def probe_inputs(batch=2):
    ids = torch.from_numpy(encode_boards([START] * batch))
    return pack_board_tokens(ids, "squares")


class AttentionCaptureTests(unittest.TestCase):
    def test_capture_does_not_change_outputs(self):
        model = tiny_model().eval()
        inputs = probe_inputs()
        (block,) = model.backbone.cores[0].layers
        with torch.no_grad():
            expected, _ = model(**inputs)
            block.attention_capture = []
            actual, _ = model(**inputs)
            block.attention_capture = None
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_zero_keys_give_uniform_attention(self):
        model = tiny_model().eval()
        (block,) = model.backbone.cores[0].layers
        with torch.no_grad():
            block.key_value.weight.zero_()
        capture = []
        block.attention_capture = capture
        with torch.no_grad():
            model(**probe_inputs())
        block.attention_capture = None
        # Zero keys make every score zero, so attention is uniform over the 81
        # valid tokens: 79 board/metadata tokens plus the act and value tokens.
        (stats,) = capture
        self.assertEqual(len(stats), 2)
        for entropy, peak in stats:
            self.assertAlmostEqual(entropy, math.log(81), places=3)
            self.assertAlmostEqual(peak, 1 / 81, places=4)


class EffectiveRankTests(unittest.TestCase):
    def test_constructed_ranks(self):
        # A diagonal matrix's singular values are its diagonal; the entropy
        # effective rank of [1, ..., 1, 0, ..., 0] with r ones is exactly r.
        for rank in (1, 2, 8):
            diagonal = torch.tensor([1.0] * rank + [0.0] * (8 - rank))
            spectrum = torch.linalg.svdvals(torch.diag(diagonal))
            self.assertAlmostEqual(effective_rank(spectrum), float(rank), places=3)


class DeadUnitTests(unittest.TestCase):
    def test_zeroed_channels_are_reported_dead(self):
        model = tiny_model().eval()
        (block,) = model.backbone.cores[0].layers
        with torch.no_grad():
            block.ffn.up_proj.weight[:8].zero_()
            block.ffn.gate_proj.weight[:8].zero_()
        probe = InternalProbe(model)
        scalars, arrays = probe.collect(model, probe_inputs())
        probe.close()
        # Channels 0-7 have both projections zeroed, so they stay exactly zero;
        # the remaining 24 channels are randomly initialized and active.
        self.assertAlmostEqual(scalars["layer_00/ffn_dead_fraction"], 8 / 32)
        self.assertTrue((arrays["ffn_channel_max_00"][:8] == 0).all())
        self.assertTrue((arrays["ffn_channel_max_00"][8:] > 1e-4).all())


class EmbeddingDriftTests(unittest.TestCase):
    def test_untrained_model_is_fully_starved_then_one_row_moves(self):
        model = tiny_model().eval()
        probe = InternalProbe(model)
        scalars, _ = probe.collect(model, probe_inputs())
        self.assertEqual(scalars["embedding/squares_starved_fraction"], 1.0)
        self.assertEqual(scalars["drift/embedding"], 0.0)
        with torch.no_grad():
            model.board_embedding.squares.weight[3] += 0.5
        scalars, _ = probe.collect(model, probe_inputs())
        probe.close()
        self.assertAlmostEqual(scalars["embedding/squares_starved_fraction"], 63 / 64)
        self.assertGreater(scalars["drift/embedding"], 0.0)


if __name__ == "__main__":
    unittest.main()
