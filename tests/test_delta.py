"""Delta-memory arithmetic, document isolation and cached/full equivalence."""

import unittest

import torch

from pondera.models.cyfa import CyclicFlowModel
from pondera.models.delta import (
    GatedDeltaModel,
    KimiDeltaModel,
    delta_recurrent,
    replay_delta_attention,
)
from pondera.models.gdn2 import GatedDeltaNet2Model
from pondera.models.kalman import KalmanDeltaModel
from pondera.models.looped import LoopedModel, RecurrentDepthModel


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

    def test_replay_order_single_decay_and_cross_loop_gradients(self):
        # With S=8, decay=1/2 and writes (v,beta)=(10,3/4),(6,1/2),(2,1/4),
        # the two-token recurrence is S' = 3*S/64 + 89/16.
        for channel_decay in (False, True):
            q = k = torch.ones(2, 1, 1)
            values = [
                torch.full((2, 1, 1), v, requires_grad=True) for v in (2.0, 6.0, 10.0)
            ]
            keys = [torch.ones(2, 1, 1, requires_grad=True) for _ in range(2)]
            strengths = [torch.full((2, 1), b, requires_grad=True) for b in (0.25, 0.5)]
            decay = torch.full(
                (2, 1, 1) if channel_decay else (2, 1), -0.6931471805599453
            )
            output, state = replay_delta_attention(
                q,
                k,
                values[2],
                decay,
                torch.full((2, 1), 0.75),
                tuple(zip(keys, values[:2], strengths)),
                (torch.tensor([0, 2]), torch.tensor([0, 2])),
                cache=torch.tensor([[[[8.0]]]]),
                output_final_state=True,
                normalize=False,
                channel_decay=channel_decay,
            )
            torch.testing.assert_close(
                output.flatten(), torch.tensor([5.9375, 5.8408203125])
            )
            torch.testing.assert_close(state.flatten(), torch.tensor([5.8408203125]))
            output.sum().backward()
            for value, expected in zip(
                values,
                (
                    [0.26171875, 0.25],
                    [0.392578125, 0.375],
                    [0.29443359375, 0.28125],
                ),
            ):
                torch.testing.assert_close(value.grad.flatten(), torch.tensor(expected))
            # For the oldest write: dS/dbeta=v-k*S; dS/dk=beta*(v-2*k*S).
            torch.testing.assert_close(
                strengths[0].grad.flatten(), torch.tensor([-5.49609375, -5.12109375])
            )
            torch.testing.assert_close(
                keys[0].grad.flatten(), torch.tensor([-3.271484375, -3.060546875])
            )

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
        for model_class, scaling in (
            (LoopedModel, False),
            (LoopedModel, True),
            (RecurrentDepthModel, False),
        ):
            for mixer in ("gdn", "kda"):
                yield model_class(
                    num_blocks=6,
                    hidden_size=32,
                    intermediate_size=48,
                    num_heads=4,
                    num_kv_heads=2,
                    vocab_size=53,
                    mixer=mixer,
                    softmax_every=3,
                    loop_repetitions=3,
                    scale_residuals=scaling,
                )
        for injection in ("decay", "orthogonal"):
            yield LoopedModel(
                num_blocks=6,
                hidden_size=32,
                intermediate_size=48,
                num_heads=4,
                num_kv_heads=2,
                vocab_size=53,
                mixer="kda",
                softmax_every=3,
                loop_repetitions=3,
                input_injection=injection,
            )
        for mixer in ("gdn", "kda"):
            yield LoopedModel(
                num_blocks=6,
                hidden_size=32,
                intermediate_size=48,
                num_heads=4,
                num_kv_heads=2,
                vocab_size=53,
                mixer=mixer,
                softmax_every=3,
                loop_repetitions=3,
                kv_replay=True,
                input_injection="decay",
            )
        for interval in (0, 2):
            yield CyclicFlowModel(
                num_blocks=2,
                hidden_size=32,
                intermediate_size=48,
                num_heads=4,
                num_kv_heads=2 if interval else 4,
                vocab_size=53,
                softmax_every=interval,
            )

        for interval in (0, 2):
            yield KalmanDeltaModel(
                num_blocks=2,
                hidden_size=32,
                intermediate_size=48,
                num_heads=4,
                num_kv_heads=2 if interval else 4,
                vocab_size=53,
                softmax_every=interval,
            )

        for interval in (0, 2):
            yield GatedDeltaNet2Model(
                num_blocks=2,
                hidden_size=32,
                intermediate_size=48,
                num_heads=4,
                num_kv_heads=2 if interval else 4,
                vocab_size=53,
                softmax_every=interval,
            )

    def test_one_loop_matches_the_untied_delta_model(self):
        config = dict(
            num_blocks=8,
            hidden_size=32,
            intermediate_size=48,
            num_heads=4,
            num_kv_heads=2,
            vocab_size=53,
            softmax_every=4,
        )
        ids = torch.randint(0, 53, (11,))
        for mixer, model_class in (("gdn", GatedDeltaModel), ("kda", KimiDeltaModel)):
            with self.subTest(mixer=mixer):
                torch.manual_seed(640)
                ordinary = model_class(**config).eval()
                torch.manual_seed(640)
                looped = LoopedModel(
                    **config, mixer=mixer, loop_repetitions=1, kv_replay=True
                ).eval()
                with torch.no_grad():
                    torch.testing.assert_close(
                        looped(ids, [0, 5, 11]),
                        ordinary(ids, [0, 5, 11]),
                        rtol=0,
                        atol=0,
                    )
                looped.loop_repetitions = 6
                self.assertEqual(
                    sum(p.numel() for p in looped.parameters()),
                    sum(p.numel() for p in ordinary.parameters()),
                )

    def test_loop_recomputation_preserves_all_gradients(self):
        import copy

        ids = torch.randint(0, 53, (11,))
        for model in self.models():
            if not isinstance(model, LoopedModel):
                continue
            recomputed = copy.deepcopy(model)
            recomputed.activation_checkpointing = True
            for network in (model, recomputed):
                network(ids, [0, 5, 11]).square().mean().backward()
            for actual, expected in zip(
                recomputed.parameters(), model.parameters(), strict=True
            ):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)

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
                torch.testing.assert_close(packed, separate, rtol=1e-5, atol=3e-6)
                for name, parameter in model.named_parameters():
                    # Repeated blocks amplify fp32 GEMM rounding when the
                    # batch is split; compare each complete parameter gradient.
                    difference = (parameter.grad - gradients[name]).norm()
                    scale = gradients[name].norm().clamp_min(1e-12)
                    self.assertLess(float(difference / scale), 2e-5, name)

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
