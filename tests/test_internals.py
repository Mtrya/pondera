import math
import unittest

import numpy as np
import torch

from pondera.chess.codec import encode_boards
from pondera.data.trace import TraceWriter
from pondera.models.policy import DepthPolicyModel, SequencePolicyModel
from pondera.training.internals import InternalProbe, effective_rank

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def tiny_model():
    torch.manual_seed(640)
    return DepthPolicyModel(
        cores=[{"layers": 1, "repetition": {"max_repetitions": 1}}],
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        dropout=0.0,
    )


def probe_inputs(batch=2):
    return {"ids": torch.from_numpy(encode_boards([START] * batch))}


def tiny_sequence(blocks=2):
    torch.manual_seed(640)
    return SequencePolicyModel(
        num_blocks=blocks,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
    )


def sequence_inputs(walk=0):
    """One trace with `walk` move/BACK pairs before the final decision."""
    writer = TraceWriter(START)
    for _ in range(walk):
        writer.move("e2e4")
        writer.back()
    writer.final("d2d4")
    return {
        "ids": torch.from_numpy(encode_boards([START])),
        "trace_ids": torch.from_numpy(np.stack([writer.tokens])),
    }


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
        # Zero keys make every score zero, so attention is uniform over the 42
        # valid tokens: 32 pieces plus 8 metadata plus the act and value tokens.
        (stats,) = capture
        self.assertEqual(len(stats), 2)
        for entropy, peak in stats:
            self.assertAlmostEqual(entropy, math.log(42), places=3)
            self.assertAlmostEqual(peak, 1 / 42, places=4)


class SequenceProbeTests(unittest.TestCase):
    def test_capture_is_output_invariant_and_reports_health(self):
        model = tiny_sequence().eval()
        inputs = sequence_inputs()
        with torch.no_grad():
            expected = model(**inputs)
            captured, output = model(**inputs, return_backbone=True, capture=True)
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        self.assertEqual(len(output.readouts), 2)

        probe = InternalProbe(model)
        scalars, arrays = probe.collect(model, inputs)
        probe.close()
        for key in (
            "layer_00/residual_rms",
            "layer_00/attention_entropy_mean",
            "layer_00/effective_rank",
            "layer_00/ffn_dead_fraction",
            "embedding/tokens_row_norm_mean",
            "embedding/tokens_starved_fraction",
            "drift/embedding",
        ):
            self.assertIn(key, scalars)
        self.assertEqual(arrays["attention_entropy"].shape, (2, 2))
        # The fresh model has not moved any embedding row yet.
        self.assertEqual(scalars["embedding/tokens_starved_fraction"], 1.0)
        self.assertEqual(scalars["drift/embedding"], 0.0)

    def test_long_sequence_attention_is_uniform_with_zero_keys(self):
        model = tiny_sequence().eval()
        for block in model.backbone.cores[0].layers:
            with torch.no_grad():
                block.key_value.weight.zero_()
        # 150 move/BACK pairs plus FINAL and its move: 302 trace tokens, long
        # enough to cross the probe's query chunk.
        inputs = sequence_inputs(walk=150)
        trace_length = inputs["trace_ids"].shape[1]
        self.assertEqual(trace_length, 302)
        probe = InternalProbe(model)
        _, arrays = probe.collect(model, inputs)
        probe.close()

        # Zero keys make attention uniform over the accessible keys. Board
        # queries see the whole B-token prefix; the j-th trace query sees the
        # prefix plus j+1 trace positions. The capture averages per query, so
        # the expected entropy and peak are hand-averaged over those counts.
        board = inputs["ids"].shape[1]
        keys = [board] * board + [board + index + 1 for index in range(trace_length)]
        expected_entropy = sum(math.log(count) for count in keys) / len(keys)
        expected_peak = sum(1 / count for count in keys) / len(keys)
        for block in range(2):
            for head in range(2):
                self.assertAlmostEqual(
                    float(arrays["attention_entropy"][block, head]),
                    expected_entropy,
                    places=3,
                )
                self.assertAlmostEqual(
                    float(arrays["attention_peak"][block, head]),
                    expected_peak,
                    places=5,
                )


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
        self.assertEqual(scalars["embedding/tokens_starved_fraction"], 1.0)
        self.assertEqual(scalars["drift/embedding"], 0.0)
        with torch.no_grad():
            model.board_embedding.tokens.weight[3] += 0.5
        scalars, _ = probe.collect(model, probe_inputs())
        probe.close()
        # The table covers every board and metadata id plus one padding row.
        self.assertEqual(model.board_embedding.tokens.weight.shape[0], 950)
        self.assertAlmostEqual(scalars["embedding/tokens_starved_fraction"], 949 / 950)
        self.assertGreater(scalars["drift/embedding"], 0.0)


if __name__ == "__main__":
    unittest.main()
