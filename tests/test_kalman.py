"""Analytic Kalman uncertainty and residual-memory updates."""

import unittest

import torch

from pondera.models.kalman import kalman_attention
from pondera.models.memory import RecurrentAttention


class KalmanTests(unittest.TestCase):
    def test_repeated_observations_average_with_the_prior(self):
        q = k = torch.ones(3, 1, 1)
        v = torch.tensor([2.0, 4.0, 8.0]).view(3, 1, 1).requires_grad_()
        output, (memory, covariance) = kalman_attention(
            q,
            k,
            v,
            torch.zeros_like(q),
            torch.zeros(3, 1),
            torch.ones(3, 1),
            RecurrentAttention.prepare([0, 3], "cpu"),
            torch.ones(1, 1),
            output_final=True,
        )
        # Unit observation variance and unit prior precision give mean sum(v)/(t+1).
        torch.testing.assert_close(output.flatten(), torch.tensor([1.0, 2.0, 3.5]))
        torch.testing.assert_close(memory.flatten(), torch.tensor([3.5]))
        torch.testing.assert_close(covariance.flatten(), torch.tensor([0.25]))
        output.sum().backward()
        torch.testing.assert_close(
            v.grad.flatten(), torch.tensor([13 / 12, 7 / 12, 1 / 4])
        )
