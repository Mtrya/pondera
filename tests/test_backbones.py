import unittest

import torch
from torch import nn

from pondera.models import (
    BaseBackbone,
    BlockConfig,
    CoreConfig,
    DepthBackbone,
    RepetitionConfig,
    SequenceBackbone,
)
from pondera.models.backbones import CausalBackbone


class BackboneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.block = BlockConfig(16, 32, 2)

    def test_multiple_repeated_cores_and_readout_gradients(self):
        cores = [
            CoreConfig(),
            CoreConfig(1, RepetitionConfig(max_repetitions=3)),
            CoreConfig(2, RepetitionConfig(max_repetitions=2)),
            CoreConfig(),
        ]
        model = DepthBackbone(self.block, cores)
        x = torch.randn(2, 5, 16, requires_grad=True)
        result = model(x, capture=True)
        self.assertEqual(len(result.readouts), 9)
        self.assertEqual(sum(len(core.layers) for core in model.cores), 5)
        self.assertEqual(
            [int(count[0, 0]) for count in result.repetitions], [1, 3, 2, 1]
        )
        external_head = nn.Linear(16, 3)
        loss = external_head(result.readouts[3].hidden).square().mean()
        loss.backward()
        self.assertGreater(float(x.grad.abs().sum()), 0)
        self.assertIsNone(model.cores[2].layers[0].query.weight.grad)
        self.assertGreater(
            float(model.cores[1].layers[0].query.weight.grad.abs().sum()), 0
        )

    def test_future_thoughts_cannot_change_prefix_or_earlier_thoughts(self):
        model = SequenceBackbone(self.block, 3)
        x = torch.randn(2, 8, 16, requires_grad=True)
        original = model(x, 4).hidden
        changed = x.detach().clone()
        changed[:, 6:] += torch.randn_like(changed[:, 6:]) * 20
        torch.testing.assert_close(model(changed, 4).hidden[:, :6], original[:, :6])
        original[:, 4, 0].sum().backward()
        self.assertEqual(float(x.grad[:, 5:].abs().sum()), 0)
        self.assertGreater(float(x.grad[:, :4].abs().sum()), 0)
        board_changed = x.detach().clone()
        board_changed[:, 3] *= -3
        self.assertGreater(
            float((model(board_changed, 4).hidden[:, 0] - original[:, 0]).abs().sum()),
            0,
        )

    def test_all_cache_policies_match_parallel_with_uneven_depth(self):
        cores = [
            CoreConfig(2, RepetitionConfig(max_repetitions=4)),
            CoreConfig(1, RepetitionConfig(max_repetitions=2)),
        ]
        x = torch.randn(2, 7, 16)
        valid = torch.tensor([[True] * 7, [True, True, False, True, True, True, True]])
        for policy in ("matched_fallback", "first", "matched_only"):
            with self.subTest(policy=policy):
                model = CausalBackbone(self.block, cores, policy).eval()
                full = model(x, 3, valid=valid)
                output = model.prefill(x[:, :3], valid=valid[:, :3])
                values = [output.hidden]
                for t in range(3, 7):
                    output = model.decode(
                        x[:, t : t + 1], output.cache, valid=valid[:, t : t + 1]
                    )
                    values.append(output.hidden)
                torch.testing.assert_close(
                    torch.cat(values, 1), full.hidden, atol=2e-6, rtol=1e-5
                )
                for expected, actual in zip((4, 2), full.repetitions):
                    torch.testing.assert_close(actual, torch.where(valid, expected, 0))

    def test_sequence_chunked_decoding_and_cache_is_persistent(self):
        model = SequenceBackbone(self.block, 2).eval()
        x = torch.randn(2, 8, 16)
        full = model(x, 3).hidden
        prefix = model.prefill(x[:, :3])
        first = model.decode(x[:, 3:5], prefix.cache)
        last = model.decode(x[:, 5:], first.cache)
        self.assertEqual(prefix.cache.length, 3)
        self.assertEqual(last.cache.length, 8)
        torch.testing.assert_close(
            torch.cat((prefix.hidden, first.hidden, last.hidden), 1), full
        )
        repeated = model.decode(x[:, 3:5], prefix.cache)
        torch.testing.assert_close(repeated.hidden, first.hidden)

    def test_padding_invariance(self):
        cores = [CoreConfig(2, RepetitionConfig(max_repetitions=3))]
        for model in (BaseBackbone(self.block, 2), DepthBackbone(self.block, cores)):
            x = torch.randn(2, 4, 16)
            expected = model(x).hidden
            padded = torch.cat((x, torch.randn(2, 3, 16)), 1)
            valid = torch.tensor([[True] * 4 + [False] * 3]).expand(2, -1)
            actual = model(padded, valid=valid).hidden
            torch.testing.assert_close(actual[:, :4], expected)
            self.assertEqual(float(actual[:, 4:].abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
