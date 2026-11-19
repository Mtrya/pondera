"""Delta-memory arithmetic, document isolation and cached/full equivalence."""

import unittest

import torch

from pondera.models.delta import GatedDeltaModel, KimiDeltaModel, delta_recurrent


class DeltaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(640)

    def test_scalar_residual_write_and_value_gradient(self):
        q = k = torch.ones(3, 1, 1)
        v = torch.tensor([[[2.0]], [[4.0]], [[8.0]]], requires_grad=True)
        g = torch.full((3, 1), -0.6931471805599453)
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

    def models(self):
        for model_class in (GatedDeltaModel, KimiDeltaModel):
            for interval in (0, 2):
                yield model_class(
                    num_blocks=2,
                    hidden_size=32,
                    intermediate_size=48,
                    num_heads=4,
                    num_kv_heads=2 if interval else 4,
                    vocab_size=53,
                    softmax_every=interval,
                )

    def test_packed_documents_match_separate_outputs_and_gradients(self):
        ids = torch.randint(0, 53, (11,))
        for model in self.models():
            with self.subTest(
                architecture=model.architecture, interval=model.softmax_every
            ):
                packed = model(ids, [0, 5, 11])
                packed.square().sum().backward()
                gradients = {
                    name: p.grad.clone() for name, p in model.named_parameters()
                }
                model.zero_grad(set_to_none=True)
                separate = torch.cat((model(ids[:5], [0, 5]), model(ids[5:], [0, 6])))
                separate.square().sum().backward()
                torch.testing.assert_close(packed, separate, rtol=1e-5, atol=1e-6)
                for name, parameter in model.named_parameters():
                    # Splitting GEMMs changes fp32 cancellation in individual
                    # entries; compare each parameter's complete gradient.
                    difference = (parameter.grad - gradients[name]).norm()
                    scale = gradients[name].norm().clamp_min(1e-12)
                    self.assertLess(float(difference / scale), 1e-5, name)

    def test_future_and_other_document_tokens_do_not_change_the_prefix(self):
        ids = torch.randint(0, 53, (13,))
        changed = ids.clone()
        changed[4:] = (changed[4:] + 1) % 53
        for model in self.models():
            with self.subTest(
                architecture=model.architecture, interval=model.softmax_every
            ):
                with torch.no_grad():
                    first = model(ids, [0, 7, 13])
                    second = model(changed, [0, 7, 13])
                torch.testing.assert_close(first[:4], second[:4], rtol=0, atol=0)

    def test_cached_decode_matches_full_forward(self):
        ids = torch.randint(0, 53, (9,))
        for model in self.models():
            with self.subTest(
                architecture=model.architecture, interval=model.softmax_every
            ):
                model.eval()
                with torch.inference_mode():
                    expected = model(ids, [0, 9])
                    prefill, cache = model.forward_cached(ids[:3], torch.tensor([0, 3]))
                    actual = [prefill]
                    for index in range(3, 9):
                        output, cache = model.forward_cached(
                            ids[index : index + 1], torch.tensor([0, index + 1]), cache
                        )
                        actual.append(output)
                torch.testing.assert_close(
                    torch.cat(actual), expected, rtol=1e-5, atol=1e-5
                )


if __name__ == "__main__":
    unittest.main()
