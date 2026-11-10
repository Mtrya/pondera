"""Bare linear attention: dual-path equivalence and boundary invariants."""

import unittest

import torch

from pondera.models.linear import (
    LinearVanillaModel,
    linear_chunked,
    linear_recurrent,
    linear_varlen_attention,
)
from pondera.models.vanilla import VanillaModel


class LinearAttentionTests(unittest.TestCase):
    def test_same_shape_as_softmax(self):
        config = dict(
            num_blocks=1,
            hidden_size=32,
            intermediate_size=48,
            num_heads=4,
            num_kv_heads=2,
            vocab_size=61,
        )
        softmax = VanillaModel(**config)
        linear = LinearVanillaModel(**config)
        self.assertEqual(linear.state_dict().keys(), softmax.state_dict().keys())
        self.assertEqual(
            linear.non_embedding_parameters(), softmax.non_embedding_parameters()
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_state_stays_fp32_under_autocast(self):
        q, k, v = (x.cuda().bfloat16() for x in (self.q, self.k, self.v))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, state = linear_chunked(q, k, v)
        reference_out, reference_state = linear_recurrent(q, k, v)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(state.dtype, torch.float32)
        torch.testing.assert_close(out, reference_out, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(state, reference_state, rtol=1e-4, atol=1e-4)

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        self.q = torch.randn(19, 2, 8)
        self.k = torch.randn(19, 2, 8)
        self.v = torch.randn(19, 2, 8)

    def test_chunked_matches_recurrence(self):
        for chunk in (1, 4, 64):
            out, state = linear_chunked(self.q, self.k, self.v, chunk)
            reference_out, reference_state = linear_recurrent(self.q, self.k, self.v)
            torch.testing.assert_close(out, reference_out, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(state, reference_state, rtol=1e-4, atol=1e-4)

    def test_future_chunk_values_cannot_erase_earlier_state(self):
        q = torch.ones(128, 1, 1)
        k = torch.ones_like(q)
        v = torch.ones_like(q)
        original, _ = linear_chunked(q, k, v)
        v[-1] = 1e10
        changed, _ = linear_chunked(q, k, v)
        torch.testing.assert_close(changed[:-1], original[:-1], rtol=0, atol=0)
        # Token 64 reads 64 preceding unit updates and its own unit update.
        self.assertEqual(changed[64].item(), 65.0)

    def test_multiple_chunks_match_recurrence_and_gradients(self):
        q, k, v = (torch.randn(129, 2, 8, requires_grad=True) for _ in range(3))
        out, state = linear_chunked(q, k, v)
        reference_out, reference_state = linear_recurrent(q, k, v)
        torch.testing.assert_close(out, reference_out, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(state, reference_state, rtol=1e-4, atol=1e-4)
        weight = torch.randn_like(out)
        actual = torch.autograd.grad(out, (q, k, v), weight)
        expected = torch.autograd.grad(reference_out, (q, k, v), weight)
        for gradient, reference in zip(actual, expected):
            torch.testing.assert_close(gradient, reference, rtol=1e-4, atol=1e-4)

    def test_state_is_zeroed_at_document_boundaries(self):
        cu = torch.tensor([0, 7, 19], dtype=torch.int32)
        packed = linear_varlen_attention(self.q, self.k, self.v, cu)
        separate = torch.cat(
            [
                linear_varlen_attention(
                    self.q[s:e],
                    self.k[s:e],
                    self.v[s:e],
                    torch.tensor([0, e - s], dtype=torch.int32),
                )
                for s, e in ((0, 7), (7, 19))
            ]
        )
        torch.testing.assert_close(packed, separate, rtol=1e-4, atol=1e-4)

    def test_cached_decode_matches_full_forward(self):
        model = LinearVanillaModel(
            num_blocks=1,
            hidden_size=32,
            intermediate_size=48,
            num_heads=4,
            num_kv_heads=2,
            vocab_size=61,
        ).eval()
        ids = torch.randint(0, 61, (11,))
        cu = torch.tensor([0, 11], dtype=torch.int32)
        with torch.no_grad():
            full = model(ids, cu)
            cache = None
            steps = []
            for index in range(11):
                logits, cache = model.forward_cached(
                    ids[index : index + 1],
                    torch.tensor([0, index + 1], dtype=torch.int32),
                    cache,
                )
                steps.append(logits)
        torch.testing.assert_close(full, torch.cat(steps), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
