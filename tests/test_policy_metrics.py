import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pondera.training.metrics import PolicyMetrics, policy_loss


class PolicyMetricsTests(unittest.TestCase):
    def test_legality_decomposition_and_saved_observations(self):
        logits = torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.8, 0.1]]).log()
        targets = torch.tensor([1, 2])
        legal = torch.tensor([[1, 2, -1], [1, 2, -1]])
        metrics = PolicyMetrics()
        metrics.update(logits, targets, legal, torch.tensor([8, 30]))
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "observations.npz"
            result = metrics.result(artifact)
            observations = np.load(artifact)
            np.testing.assert_allclose(
                observations["loss"],
                observations["legal_loss"] + observations["legality_penalty"],
                atol=1e-6,
            )
            self.assertAlmostEqual(observations["loss"][0], -math.log(0.3), places=6)
            self.assertAlmostEqual(
                observations["legal_loss"][0], -math.log(0.75), places=6
            )
            self.assertEqual(result["top1"], 0)
            self.assertEqual(result["legal_top1"], 0.5)
            self.assertEqual(result["legal_top5"], 1)
            self.assertEqual(result["legal_mrr"], 0.75)
            self.assertEqual(result["illegal_argmax"], 0.5)
            self.assertAlmostEqual(result["ece"], 0.7, places=6)
            self.assertEqual(result["pieces_2_10/positions"], 1)
            self.assertEqual(result["pieces_11_20/positions"], 0)
            self.assertTrue(all(math.isfinite(value) for value in result.values()))

    def test_legal_loss_excludes_illegal_action_gradients(self):
        logits = torch.tensor([[3.0, 1.0, 2.0]], requires_grad=True)
        targets = torch.tensor([1])
        legal = torch.tensor([[1, 2, -1]])
        loss = policy_loss(logits, targets, legal)
        expected = torch.nn.functional.cross_entropy(logits[:, 1:], torch.tensor([0]))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(logits.grad[0, 0], 0)
        self.assertLess(logits.grad[0, 1], 0)
        self.assertGreater(logits.grad[0, 2], 0)
