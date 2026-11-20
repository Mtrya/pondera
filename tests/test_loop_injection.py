"""Input preservation and differentiable projection across recurrent depth."""

import unittest

import torch

from pondera.models.looped import InputInjection, LoopedModel


class InputInjectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_projection_removes_parallel_carry_and_preserves_both_gradients(self):
        injection = InputInjection(2, orthogonal=True)
        state = torch.tensor([[3.0, 4.0]], requires_grad=True)
        anchor = torch.tensor([[2.0, 0.0]], requires_grad=True)
        output = injection(state, (anchor, torch.ones(2)))
        # Projection on the x axis removes (3, 0), then injects (2, 0).
        torch.testing.assert_close(
            output, torch.tensor([[2.0, 4.0]]), atol=2e-6, rtol=0
        )
        output.sum().backward()
        # At q=(2,0), d(sum output)/dq=(1,-(3+4)/2+1).
        torch.testing.assert_close(
            state.grad, torch.tensor([[0.0, 1.0]]), atol=2e-6, rtol=0
        )
        torch.testing.assert_close(
            anchor.grad, torch.tensor([[1.0, -2.5]]), atol=2e-6, rtol=0
        )

    def test_zero_input_retains_decayed_state_and_finite_gradient(self):
        injection = InputInjection(2, orthogonal=True)
        state = torch.tensor([[3.0, -4.0]], requires_grad=True)
        anchor = torch.zeros_like(state, requires_grad=True)
        output = injection(state, (anchor, torch.tensor([0.5, 0.25])))
        torch.testing.assert_close(output, torch.tensor([[1.5, -1.0]]), atol=0, rtol=0)
        output.sum().backward()
        self.assertTrue(torch.isfinite(anchor.grad).all())

    def test_input_and_state_gradients_match_finite_differences(self):
        torch.manual_seed(640)
        injection = InputInjection(4, orthogonal=True).double()
        state, embedded = (
            torch.randn(2, 4, dtype=torch.float64, requires_grad=True) for _ in range(2)
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda s, e: injection(s, injection.prepare(e)), (state, embedded)
            )
        )

    def test_zero_initial_state_makes_projection_control_identical_at_one_loop(self):
        settings = dict(
            num_blocks=6,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            num_kv_heads=1,
            vocab_size=31,
            loop_repetitions=1,
        )
        torch.manual_seed(640)
        decay = LoopedModel(**settings, input_injection="decay")
        torch.manual_seed(640)
        orthogonal = LoopedModel(**settings, input_injection="orthogonal")
        ids = torch.arange(9)
        for model in (decay, orthogonal):
            model(ids, [0, 4, 9]).square().sum().backward()
        for actual, expected in zip(
            orthogonal.parameters(), decay.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
