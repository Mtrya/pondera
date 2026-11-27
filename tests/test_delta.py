"""Delta-memory arithmetic, document isolation and cached/full equivalence."""

import unittest

import torch

from pondera.models.delta import delta_recurrent


class DeltaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(640)

    def test_scalar_residual_write_and_value_gradient(self):
        q = k = torch.ones(3, 1, 1)
        v = torch.tensor([[[2.0]], [[4.0]], [[8.0]]], requires_grad=True)
        g = torch.full((3, 1, 1), -0.6931471805599453)
        beta = torch.full((3, 1), 0.5)
        output, state = delta_recurrent(q, k, v, g, beta)
        # Half decay followed by a half residual correction gives a 1/4 carry.
        torch.testing.assert_close(output.flatten(), torch.tensor([1.0, 2.25, 4.5625]))
        torch.testing.assert_close(state, torch.tensor([[[4.5625]]]))
        output.sum().backward()
        torch.testing.assert_close(
            v.grad.flatten(), torch.tensor([0.65625, 0.625, 0.5])
        )

    def test_channel_decay_keeps_the_other_key_channel(self):
        q = torch.tensor([[[1.4142135623730951, 0.0]], [[0.0, 1.4142135623730951]]])
        k = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        v = torch.tensor([[[2.0]], [[8.0]]])
        g = torch.tensor([[[-0.6931471805599453, 0.0]]] * 2)
        output, state = delta_recurrent(q, k, v, g, torch.full((2, 1), 0.5))
        # The first association halves while the orthogonal key writes four.
        torch.testing.assert_close(output.flatten(), torch.tensor([1.0, 4.0]))
        torch.testing.assert_close(state, torch.tensor([[[0.5], [4.0]]]))
