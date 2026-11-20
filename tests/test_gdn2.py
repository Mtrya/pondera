"""GDN-2's scalar-gate reduction and erase projection contract."""

import unittest

import torch

from pondera.models.delta import GatedDeltaAttention, delta_attention
from pondera.models.gdn2 import gdn2_attention, guard_erase


class GatedDeltaNet2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(640)

    def test_shared_scalar_erase_write_recovers_kda_and_its_gradients(self):
        inputs = [
            torch.randn(11, 2, 8),
            torch.randn(11, 2, 8),
            torch.randn(11, 2, 8),
            -torch.rand(11, 2, 8) * 0.2 - 0.01,
            torch.rand(11, 2),
            torch.randn(2, 2, 8, 8),
        ]
        expected_inputs = [x.clone().requires_grad_() for x in inputs]
        actual_inputs = [x.clone().requires_grad_() for x in inputs]
        prepared = GatedDeltaAttention.prepare([0, 4, 11], "cpu")
        q, k, v, g, beta, initial = expected_inputs
        expected, expected_state = delta_attention(
            q, k, v, g, beta, prepared, initial, True, channel_decay=True
        )
        q, k, v, g, beta, initial = actual_inputs
        actual, actual_state = gdn2_attention(
            q,
            k,
            v,
            g,
            beta.unsqueeze(-1).expand_as(k),
            beta.unsqueeze(-1).expand_as(v),
            prepared,
            initial,
            True,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual_state, expected_state, rtol=2e-5, atol=2e-6)
        cotangent, terminal = torch.randn_like(actual), torch.randn_like(actual_state)
        for output, state in ((actual, actual_state), (expected, expected_state)):
            ((output * cotangent).sum() + (state * terminal).sum()).backward()
        for actual_input, expected_input in zip(actual_inputs, expected_inputs):
            torch.testing.assert_close(
                actual_input.grad, expected_input.grad, rtol=3e-5, atol=3e-6
            )

    def test_guard_contracts_nonnormal_example_with_identity_erase_gradient(self):
        key = torch.tensor([[2**-0.5, 2**-0.5]], requires_grad=True)
        erase = torch.tensor([[0.0, 1.0]], requires_grad=True)
        decay = torch.full_like(key, -0.001, requires_grad=True)
        projected = guard_erase(key, erase, decay)
        # The 2D guard equation gives mean .5 and contrast factor .0763033.
        torch.testing.assert_close(projected, torch.tensor([[0.46184835, 0.53815162]]))
        upstream = torch.tensor([[0.25, -0.75]])
        gradients = torch.autograd.grad(
            projected, (key, erase, decay), upstream, allow_unused=True
        )
        self.assertIsNone(gradients[0])
        torch.testing.assert_close(gradients[1], upstream, rtol=0, atol=0)
        self.assertIsNone(gradients[2])

    def test_safe_scalar_and_zero_key_leave_erase_unchanged(self):
        key = torch.randn(4, 8)
        key = key / key.norm(dim=-1, keepdim=True)
        key[0] = 0
        erase = torch.rand(4, 8)
        erase[1] = 0.5
        decay = torch.full_like(key, -1.0)
        decay[:2] = 0
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                actual = guard_erase(key.to(dtype), erase.to(dtype), decay)
                torch.testing.assert_close(actual, erase.to(dtype), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
