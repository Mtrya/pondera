"""Depth attention checked against geometric weights and ordinary softmax attention."""

import unittest

import torch
from torch.nn import functional as F

from pondera.models.looped import LoopAttentionResidual


class LoopAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_zero_query_has_geometric_weights_including_the_initial_state(self):
        connection = LoopAttentionResidual(2, 0.5)
        states = torch.tensor(
            [[[2.0, 0.0]], [[0.0, 4.0]], [[6.0, 2.0]]], requires_grad=True
        )
        memory = connection.start(states[0])
        for state in states[1:]:
            output, memory = connection(state, memory)
        # Decay .5 gives normalized weights 1/7, 2/7, 4/7 after two updates.
        torch.testing.assert_close(output, torch.tensor([[3.714285714, 2.285714286]]))
        output.sum().backward()
        torch.testing.assert_close(
            states.grad,
            torch.tensor(
                [
                    [[0.142857143, 0.142857143]],
                    [[0.285714286, 0.285714286]],
                    [[0.571428571, 0.571428571]],
                ]
            ),
        )

    def test_outputs_and_all_gradients_match_softmax_attention_over_depth(self):
        torch.manual_seed(640)
        connection = LoopAttentionResidual(5, 0.61).double()
        with torch.no_grad():
            connection.query.copy_(torch.linspace(-0.3, 0.3, 5))
        states = torch.randn(4, 3, 5, dtype=torch.float64, requires_grad=True)
        memory = connection.start(states[0])
        for state in states[1:]:
            output, memory = connection(state, memory)
        keys = F.rms_norm(states, (5,), eps=1e-6).transpose(0, 1)
        # The equivalent dense attention has an additive log-decay position bias.
        bias = torch.arange(3, -1, -1) * F.logsigmoid(connection.decay_logit)
        expected = F.scaled_dot_product_attention(
            connection.query.expand(3, 1, 5),
            keys,
            states.transpose(0, 1),
            attn_mask=bias.expand(3, 1, 4),
            scale=1.0,
        ).squeeze(1)
        torch.testing.assert_close(output, expected)
        inputs = (states, connection.query, connection.decay_logit)
        actual_gradients = torch.autograd.grad(output.square().sum(), inputs)
        expected_gradients = torch.autograd.grad(expected.square().sum(), inputs)
        for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
            torch.testing.assert_close(actual, reference)

    def test_repeating_the_same_state_preserves_it_with_extreme_scores(self):
        connection = LoopAttentionResidual(3, 0.9999)
        with torch.no_grad():
            connection.query.fill_(1e6)
            state = torch.tensor([[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]])
            memory = connection.start(state)
            for _ in range(128):
                output, memory = connection(state, memory)
                torch.testing.assert_close(output, state, rtol=2e-6, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
