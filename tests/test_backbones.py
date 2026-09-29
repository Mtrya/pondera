import unittest

import torch
from torch import nn

from pondera.models import (
    BaseBackbone,
    BlockConfig,
    CombinedBackbone,
    CoreConfig,
    DepthBackbone,
    RepetitionConfig,
    SequenceBackbone,
)


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
        for model in (
            SequenceBackbone(self.block, 3),
            CombinedBackbone(
                self.block, [CoreConfig(2, RepetitionConfig(max_repetitions=3))]
            ),
        ):
            with self.subTest(model=type(model).__name__):
                x = torch.randn(2, 8, 16, requires_grad=True)
                original = model(x, 4).hidden
                changed = x.detach().clone()
                changed[:, 6:] += torch.randn_like(changed[:, 6:]) * 20
                torch.testing.assert_close(
                    model(changed, 4).hidden[:, :6], original[:, :6]
                )
                original[:, 4, 0].sum().backward()
                self.assertEqual(float(x.grad[:, 5:].abs().sum()), 0)
                self.assertGreater(float(x.grad[:, :4].abs().sum()), 0)
                board_changed = x.detach().clone()
                board_changed[:, 3] *= -3
                self.assertGreater(
                    float(
                        (model(board_changed, 4).hidden[:, 0] - original[:, 0])
                        .abs()
                        .sum()
                    ),
                    0,
                )

    def test_all_cache_policies_match_parallel_with_uneven_depth(self):
        cores = [
            CoreConfig(2, RepetitionConfig(max_repetitions=4), True, True),
            CoreConfig(1, RepetitionConfig(max_repetitions=2)),
        ]
        x = torch.randn(2, 7, 16)
        valid = torch.tensor([[True] * 7, [True, True, False, True, True, True, True]])
        limits = [
            torch.tensor([[2, 4, 1, 2, 4, 3, 1], [4, 1, 3, 2, 1, 4, 3]]),
            torch.tensor([[2, 1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2, 1]]),
        ]
        for policy in ("matched_fallback", "first", "matched_only"):
            with self.subTest(policy=policy):
                model = CombinedBackbone(self.block, cores, policy).eval()
                full = model(x, 3, valid=valid, repetition_limits=limits)
                output = model.prefill(
                    x[:, :3],
                    valid=valid[:, :3],
                    repetition_limits=[v[:, :3] for v in limits],
                )
                values = [output.hidden]
                for t in range(3, 7):
                    output = model.decode(
                        x[:, t : t + 1],
                        output.cache,
                        valid=valid[:, t : t + 1],
                        repetition_limits=[v[:, t : t + 1] for v in limits],
                    )
                    values.append(output.hidden)
                torch.testing.assert_close(
                    torch.cat(values, 1), full.hidden, atol=2e-6, rtol=1e-5
                )
                for expected, actual in zip(limits, full.repetitions):
                    torch.testing.assert_close(actual, expected * valid)

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

    def test_learned_shared_board_decision_does_not_share_thought_decisions(self):
        config = RepetitionConfig(mode="learned", max_repetitions=4, scope="shared")
        model = CombinedBackbone(self.block, [CoreConfig(1, config)]).eval()
        with torch.no_grad():
            for module in model.cores[0].layers.modules():
                if isinstance(module, nn.Linear):
                    module.weight.zero_()
                    if module.bias is not None:
                        module.bias.zero_()
            head = model.cores[0].controller.head
            head.weight.zero_()
            head.weight[0, 0] = 1
            head.bias.zero_()
        x = torch.ones(1, 4, 16)
        x[:, 1] *= -1
        x[:, 2] *= -1
        output = model(x, 2)
        self.assertEqual(output.repetitions[0].tolist(), [[1, 1, 4, 1]])
        prefix = model.prefill(x[:, :2])
        a = model.decode(x[:, 2:3], prefix.cache)
        b = model.decode(x[:, 3:], a.cache)
        torch.testing.assert_close(
            torch.cat((prefix.hidden, a.hidden, b.hidden), 1), output.hidden
        )
        sum(
            readout.logits[readout.active].sum() for readout in output.controllers
        ).backward()
        self.assertGreater(float(head.weight.grad.abs().sum()), 0)

    def test_convergence_obeys_minimum_and_skips_later_work(self):
        config = RepetitionConfig(
            mode="convergence", min_repetitions=2, max_repetitions=5, tolerance=0
        )
        model = DepthBackbone(self.block, [CoreConfig(1, config)])
        with torch.no_grad():
            for module in model.cores[0].layers.modules():
                if isinstance(module, nn.Linear):
                    module.weight.zero_()
                    if module.bias is not None:
                        module.bias.zero_()
        calls = []
        handle = (
            model.cores[0]
            .layers[0]
            .ffn.register_forward_hook(
                lambda module, args, result: calls.append(args[0].shape)
            )
        )
        try:
            output = model(torch.randn(2, 6, 16))
        finally:
            handle.remove()
        self.assertEqual(len(calls), 2)
        self.assertTrue(bool((output.repetitions[0] == 2).all()))

    def test_individual_exit_reduces_ffn_query_count(self):
        model = DepthBackbone(
            self.block, [CoreConfig(1, RepetitionConfig(max_repetitions=4))]
        )
        shapes = []
        handle = (
            model.cores[0]
            .layers[0]
            .ffn.register_forward_hook(
                lambda module, args, result: shapes.append(args[0].shape[1])
            )
        )
        try:
            limits = torch.tensor([[1, 2, 4, 1]])
            output = model(torch.randn(1, 4, 16), repetition_limits=[limits])
        finally:
            handle.remove()
        self.assertEqual(shapes, [4, 2, 1, 1])
        torch.testing.assert_close(output.repetitions[0], limits)

    def test_padding_invariance_and_gradient_through_reinjection(self):
        cores = [CoreConfig(2, RepetitionConfig(max_repetitions=3), True, True)]
        for model in (BaseBackbone(self.block, 2), DepthBackbone(self.block, cores)):
            x = torch.randn(2, 4, 16, requires_grad=True)
            expected = model(x).hidden
            padded = torch.cat((x, torch.randn(2, 3, 16)), 1)
            valid = torch.tensor([[True] * 4 + [False] * 3]).expand(2, -1)
            actual = model(padded, valid=valid).hidden
            torch.testing.assert_close(actual[:, :4], expected)
            self.assertEqual(float(actual[:, 4:].abs().sum()), 0)
            expected[:, :, 0].sum().backward()
            if isinstance(model, DepthBackbone):
                self.assertGreater(
                    float(model.cores[0].reinjection.weight.grad.abs().sum()), 0
                )
                self.assertGreater(
                    float(model.cores[0].iteration.weight.grad.abs().sum()), 0
                )


if __name__ == "__main__":
    unittest.main()
