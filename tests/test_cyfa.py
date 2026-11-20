"""Cyclic memory arithmetic, independent documents and streaming state."""

import unittest

import torch

from pondera.models.cyfa import _Clock, _pack_segments, _rotation, cyclic_flow_attention
from pondera.models.delta import GatedDeltaAttention


class CyFATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(640)

    def test_segment_padding_and_gradients_exclude_unselected_tokens(self):
        # Noncontiguous rows 1--3 and 7--8; padding must not read or train row 0.
        vector = torch.arange(36.0).view(9, 4)[:, ::2].requires_grad_()
        gate = torch.arange(9.0).requires_grad_()
        packed, packed_gate = _pack_segments((vector, gate), [(0, 1, 3), (1, 7, 2)], 4)
        torch.testing.assert_close(
            packed,
            torch.tensor(
                [
                    [[4, 6], [8, 10], [12, 14], [0, 0]],
                    [[28, 30], [32, 34], [0, 0], [0, 0]],
                ],
                dtype=torch.float32,
            ),
        )
        torch.testing.assert_close(
            packed_gate, torch.tensor([[1, 2, 3, 0], [7, 8, 0, 0.0]])
        )
        (packed.sum() + packed_gate.sum()).backward()
        selected = torch.tensor([0, 1, 1, 1, 0, 0, 0, 1, 1.0])
        torch.testing.assert_close(vector.grad, selected[:, None].expand(9, 2))
        torch.testing.assert_close(gate.grad, selected)

    def inputs(self, length=13):
        return [
            *[torch.randn(length, 2, 4).requires_grad_() for _ in range(3)],
            (-0.1 * torch.rand(length, 2)).requires_grad_(),
            torch.rand(length, 2).requires_grad_(),
            torch.rand(length, 2).requires_grad_(),
            (
                torch.eye(3).expand(2, -1, -1) + 0.1 * torch.randn(2, 3, 3)
            ).requires_grad_(),
        ]

    def run_memory(self, inputs, boundaries, **kwargs):
        prepared = GatedDeltaAttention.prepare(boundaries, "cpu")
        return cyclic_flow_attention(*inputs, prepared, **kwargs)

    def test_uniform_readout_and_value_gradient(self):
        q = k = torch.zeros(4, 1, 2)
        v = torch.tensor([2.0, 4.0, 8.0, 16.0]).view(4, 1, 1).requires_grad_()
        g = torch.zeros(4, 1)
        delta = beta = torch.ones(4, 1)
        readout = torch.eye(3).unsqueeze(0)
        output, _ = self.run_memory([q, k, v, g, delta, beta, readout], [0, 4])
        # Zero content scores uniformly read three active slots: cumulative v/3.
        torch.testing.assert_close(
            output.flatten(), torch.tensor([2 / 3, 2.0, 14 / 3, 10.0])
        )
        output.sum().backward()
        torch.testing.assert_close(
            v.grad.flatten(), torch.tensor([4 / 3, 1.0, 2 / 3, 1 / 3])
        )

    def test_packed_outputs_and_all_gradients_equal_separate_documents(self):
        inputs = self.inputs()
        output, _ = self.run_memory(inputs, [0, 5, 13])
        output.square().sum().backward()
        gradients = [value.grad.clone() for value in inputs]
        for value in inputs:
            value.grad = None
        pieces = []
        for start, end in ((0, 5), (5, 13)):
            part = [value[start:end] for value in inputs[:-1]] + inputs[-1:]
            pieces.append(self.run_memory(part, [0, end - start])[0])
        separate = torch.cat(pieces)
        separate.square().sum().backward()
        torch.testing.assert_close(output, separate)
        for value, expected in zip(inputs, gradients, strict=True):
            torch.testing.assert_close(value.grad, expected, rtol=2e-5, atol=1e-5)

    def test_streaming_preserves_both_memories_and_clock(self):
        inputs = self.inputs()
        with torch.inference_mode():
            expected, expected_state = self.run_memory(
                inputs, [0, 13], output_final=True
            )
            pieces, state = [], None
            for start, end in ((0, 4), (4, 5), (5, 9), (9, 13)):
                part = [value[start:end] for value in inputs[:-1]] + inputs[-1:]
                output, state = self.run_memory(
                    part, [0, end - start], initial=state, output_final=True
                )
                pieces.append(output)
            torch.testing.assert_close(
                torch.cat(pieces), expected, rtol=1e-5, atol=1e-5
            )
            for actual, reference in zip(state, expected_state, strict=True):
                torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)

    def test_future_tokens_do_not_change_prefix(self):
        inputs = self.inputs()
        changed = [value.detach().clone() for value in inputs]
        for value in changed[:-1]:
            value[7:] = 0.0
        expected, _ = self.run_memory(inputs, [0, 13])
        actual, _ = self.run_memory(changed, [0, 13])
        torch.testing.assert_close(actual[:7], expected[:7], rtol=0, atol=0)

    def test_long_clock_phase_and_streaming_against_float64(self):
        delta = torch.rand(131072, 2)
        delta[:, 1] *= 1e-6
        prepared = GatedDeltaAttention.prepare([0, len(delta)], "cpu")
        clock, state = _Clock.apply(delta, prepared, None, 63)
        truth = delta.double().cumsum(0).remainder(63)
        frequencies = torch.arange(32).double()
        angle = truth[..., None] * frequencies * (2 * torch.pi / 63)
        cosine, sine = _rotation(clock, 64)
        torch.testing.assert_close(cosine.double(), angle.cos(), atol=7e-6, rtol=0)
        torch.testing.assert_close(sine.double(), angle.sin(), atol=7e-6, rtol=0)
        split = len(delta) - 4096
        _, streaming = _Clock.apply(
            delta[:split], GatedDeltaAttention.prepare([0, split], "cpu"), None, 63
        )
        one = GatedDeltaAttention.prepare([0, 1], "cpu")
        tail = []
        for increment in delta[split:]:
            part, streaming = _Clock.apply(increment[None], one, streaming, 63)
            tail.append(part)
        torch.testing.assert_close(torch.cat(tail), clock[split:], rtol=0, atol=0)
        torch.testing.assert_close(streaming, state, rtol=0, atol=0)

    def test_empty_memory_output_is_invariant_to_initial_clock(self):
        inputs = self.inputs(81)
        expected, _ = self.run_memory(inputs, [0, 81])
        initial = (
            torch.zeros(1, 2, 4, 4),
            torch.zeros(1, 2, 4, 4),
            torch.tensor([[2.99, 0.125]]).double().mul(1 << 44).long(),
        )
        actual, _ = self.run_memory(inputs, [0, 81], initial=initial)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
