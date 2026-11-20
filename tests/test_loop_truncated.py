"""Truncated recurrence preserves the forward computation and learns its suffix."""

import copy
import unittest

import torch

from pondera.models.looped import LoopedModel, RecurrentDepthModel


class TruncatedLoopTests(unittest.TestCase):
    def models(self):
        config = dict(
            num_blocks=6,
            hidden_size=32,
            intermediate_size=48,
            num_heads=4,
            num_kv_heads=2,
            vocab_size=53,
            loop_repetitions=4,
        )
        yield RecurrentDepthModel(**config)
        yield LoopedModel(**config, input_injection="decay")
        yield LoopedModel(
            **config,
            mixer="kda",
            softmax_every=3,
            input_injection="decay",
            kv_replay=True,
            attention_residual_decay=0.55,
        )

    def test_truncation_preserves_forward_and_prelude_gradients(self):
        torch.manual_seed(640)
        ids = torch.randint(0, 53, (11,))
        for model in self.models():
            for amp in (False, True):
                with self.subTest(architecture=model.architecture, amp=amp):
                    model.backprop_repetitions = 2
                    model.zero_grad(set_to_none=True)
                    recorded = []
                    hook = model.blocks[2].norm_attention.register_forward_hook(
                        lambda module, inputs, output: recorded.append(
                            output.requires_grad
                        )
                    )
                    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
                        actual = model(ids, [0, 5, 11])
                    hook.remove()
                    # The four shared executions keep only the final two graphs.
                    self.assertEqual(recorded, [False, False, True, True])
                    actual.float().square().mean().backward()
                    for name, parameter in model.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        self.assertGreater(parameter.grad.norm().item(), 0, name)
                    model.eval()
                    with (
                        torch.no_grad(),
                        torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp),
                    ):
                        expected = model(ids, [0, 5, 11])
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    model.train()

    def test_full_horizon_recovers_all_parameter_gradients(self):
        torch.manual_seed(640)
        ids = torch.randint(0, 53, (11,))
        for model in self.models():
            capped = copy.deepcopy(model)
            capped.backprop_repetitions = model.loop_repetitions
            for network in (model, capped):
                network(ids, [0, 5, 11]).square().mean().backward()
            for actual, expected in zip(
                capped.parameters(), model.parameters(), strict=True
            ):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)

    def test_checkpoint_recomputation_preserves_truncated_gradients(self):
        torch.manual_seed(640)
        ids = torch.randint(0, 53, (11,))
        for model in self.models():
            model.backprop_repetitions = 2
            recomputed = copy.deepcopy(model)
            recomputed.activation_checkpointing = True
            for network in (model, recomputed):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    loss = network(ids, [0, 5, 11]).float().square().mean()
                loss.backward()
            for actual, expected in zip(
                recomputed.parameters(), model.parameters(), strict=True
            ):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
