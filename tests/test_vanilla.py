"""Vanilla model invariants: causality, bounded FFN, packed/eager equivalence."""

import unittest
from importlib.util import find_spec

import torch

from pondera.models.linear import LinearVanillaModel
from pondera.models.looped import LoopedModel
from pondera.models.vanilla import (
    VanillaModel,
    eager_varlen_attention,
    situ_glu,
    varlen_attention,
)

MODEL = dict(
    num_blocks=2,
    hidden_size=32,
    intermediate_size=48,
    num_heads=4,
    num_kv_heads=2,
    vocab_size=61,
)


def packed_ids():
    return torch.randint(0, 61, (11,)), torch.tensor([0, 5, 11], dtype=torch.int32)


class VanillaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = VanillaModel(**MODEL).eval()

    def test_looped_depth_shares_weights_and_cached_generation_matches(self):
        config = {**MODEL, "num_blocks": 6}
        ordinary = VanillaModel(**config).eval()
        looped = LoopedModel(**config, loop_repetitions=1).eval()
        looped.load_state_dict(ordinary.state_dict())
        ids, cu = packed_ids()
        with torch.no_grad():
            torch.testing.assert_close(
                looped(ids, cu), ordinary(ids, cu), rtol=0, atol=0
            )
            looped.loop_repetitions = 3
            self.assertEqual(
                sum(p.numel() for p in looped.parameters()),
                sum(p.numel() for p in ordinary.parameters()),
            )
            full = looped(ids[:5], [0, 5])
            _, caches = looped.forward_cached(ids[:4], torch.tensor([0, 4]))
            last, _ = looped.forward_cached(ids[4:5], torch.tensor([0, 5]), caches)
            torch.testing.assert_close(last[-1], full[-1], rtol=1e-5, atol=1e-6)
            packed = looped(ids, cu)
            torch.testing.assert_close(packed[:5], full, rtol=1e-6, atol=1e-6)

    def test_later_documents_and_future_tokens_cannot_change_earlier_logits(self):
        ids, cu = packed_ids()
        with torch.no_grad():
            base = self.model(ids, cu)
            changed = ids.clone()
            changed[5:] = torch.randint(0, 61, (6,))
            altered = self.model(changed, cu)
        torch.testing.assert_close(base[:5], altered[:5], rtol=0, atol=0)
        changed = ids.clone()
        changed[7] = (changed[7] + 1) % 61
        with torch.no_grad():
            altered = self.model(changed, cu)
        torch.testing.assert_close(base[:7], altered[:7], rtol=0, atol=0)

    def test_packed_forward_equals_per_document_forward(self):
        ids, cu = packed_ids()
        with torch.no_grad():
            packed = self.model(ids, cu)
            separate = torch.cat(
                [
                    self.model(ids[s:e], torch.tensor([0, e - s], dtype=torch.int32))
                    for s, e in ((0, 5), (5, 11))
                ]
            )
        torch.testing.assert_close(packed, separate, rtol=1e-6, atol=1e-6)

    def test_activation_checkpointing_preserves_logits_and_gradients(self):
        ids, cu = packed_ids()
        labels = torch.randint(0, MODEL["vocab_size"], ids.shape)
        for model_class in (VanillaModel, LinearVanillaModel):
            with self.subTest(architecture=model_class.architecture):
                original = model_class(**MODEL).train()
                recomputed = model_class(**MODEL).train()
                recomputed.load_state_dict(original.state_dict())
                recomputed.activation_checkpointing = True

                def forward_with_saved_elements(model):
                    saved = []

                    def pack(tensor):
                        saved.append(tensor.numel())
                        return tensor

                    with torch.autograd.graph.saved_tensors_hooks(
                        pack, lambda tensor: tensor
                    ):
                        logits = model(ids, cu)
                    return logits, sum(saved)

                logits, original_saved = forward_with_saved_elements(original)
                checkpointed_logits, checkpointed_saved = forward_with_saved_elements(
                    recomputed
                )
                self.assertLess(checkpointed_saved, original_saved)
                torch.testing.assert_close(logits, checkpointed_logits, rtol=0, atol=0)
                torch.nn.functional.cross_entropy(logits, labels).backward()
                torch.nn.functional.cross_entropy(
                    checkpointed_logits, labels
                ).backward()
                for parameter, checkpointed in zip(
                    original.parameters(), recomputed.parameters()
                ):
                    torch.testing.assert_close(
                        parameter.grad, checkpointed.grad, rtol=0, atol=0
                    )

    def test_situ_glu_output_is_elementwise_bounded(self):
        gate = torch.randn(64, 8) * 100
        up = torch.randn(64, 8) * 100
        # |4·tanh(g/4)| ≤ 4, |σ(g)| ≤ 1, |25·tanh(u/25)| ≤ 25 → |out| ≤ 100.
        self.assertLessEqual(situ_glu(gate, up).abs().max().item(), 100.0)

    def test_cached_incremental_decode_matches_full_forward(self):
        ids, _ = packed_ids()
        cu = torch.tensor([0, 11], dtype=torch.int32)
        with torch.no_grad():
            full = self.model(ids, cu)
            caches = None
            steps = []
            for index in range(11):
                logits, caches = self.model.forward_cached(
                    ids[index : index + 1],
                    torch.tensor([0, index + 1], dtype=torch.int32),
                    caches,
                )
                steps.append(logits)
        torch.testing.assert_close(full, torch.cat(steps), rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "fused varlen needs CUDA")
    def test_fused_varlen_matches_eager_reference(self):
        torch.manual_seed(3)
        q = torch.randn(11, 4, 16, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(11, 2, 16, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(11, 2, 16, device="cuda", dtype=torch.bfloat16)
        cu = torch.tensor([0, 5, 11], dtype=torch.int32, device="cuda")
        fused = varlen_attention(q, k, v, cu)
        reference = eager_varlen_attention(q, k, v, cu)
        self.assertLessEqual((fused - reference).abs().max().item(), 1e-3)


class NPUAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if find_spec("torch_npu") is None:
            raise unittest.SkipTest("fused varlen needs torch_npu")
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise unittest.SkipTest("fused varlen needs an NPU")
        cls.device = torch.device("npu:0")
        torch.npu.set_device(cls.device)

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(640)

    def test_fused_gqa_forward_gradients_and_isolation(self):
        cu = torch.tensor([0, 1, 4, 21, 150], dtype=torch.int32)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                rounded = [
                    torch.randn(150, heads, 64).to(dtype) for heads in (16, 4, 4)
                ]
                reference = [value.float().requires_grad_() for value in rounded]
                inputs = [value.to(self.device).requires_grad_() for value in rounded]
                gradient = torch.randn(150, 16, 64).to(dtype)
                # A cached mask first created in inference must also work in training.
                with torch.inference_mode():
                    varlen_attention(*inputs, cu)
                expected = eager_varlen_attention(*reference, cu)
                expected.backward(gradient.float())
                actual = varlen_attention(*inputs, cu)
                actual.backward(gradient.to(self.device))
                for measured, target in (
                    (actual, expected),
                    *((value.grad, ref.grad) for value, ref in zip(inputs, reference)),
                ):
                    difference = measured.detach().float().cpu() - target.detach()
                    self.assertLess(
                        (difference.norm() / target.detach().norm()).item(), 0.02
                    )

                with torch.no_grad():
                    changed = [value.detach().clone() for value in inputs]
                    changed[1][-1] += 10
                    changed[2][-1] -= 10
                    altered = varlen_attention(*changed, cu)
                    torch.testing.assert_close(
                        actual[:-1], altered[:-1], rtol=0, atol=0
                    )
                    changed = [value.detach().clone() for value in inputs]
                    changed[1][:21] += 10
                    changed[2][:21] -= 10
                    altered = varlen_attention(*changed, cu)
                    torch.testing.assert_close(
                        actual[21:], altered[21:], rtol=0, atol=0
                    )

    def test_document_split_preserves_outputs_and_value_gradients(self):
        # 1023 singleton documents, then lengths 3, 2, 1 across the split.
        cu = torch.tensor([*range(1024), 1026, 1028, 1029], dtype=torch.int32)
        q = torch.zeros(1029, 4, 64, dtype=torch.bfloat16, device=self.device)
        k = torch.zeros(1029, 2, 64, dtype=torch.bfloat16, device=self.device)
        v = torch.zeros_like(k)
        v[-6:] = torch.tensor([2, 4, 6, 8, 12, 20], device=self.device)[:, None, None]
        v.requires_grad_()
        actual = varlen_attention(q, k, v, cu)
        # Zero logits give uniform causal prefix averages within each document.
        expected = torch.zeros_like(q)
        expected[-6:] = torch.tensor([2, 3, 4, 8, 10, 20], device=self.device)[
            :, None, None
        ]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        # Each KV head serves two query heads; sum the causal averaging weights.
        expected_gradient = torch.full_like(v, 2)
        expected_gradient[-6:] = torch.tensor(
            [11 / 3, 5 / 3, 2 / 3, 3, 1, 2], device=self.device
        )[:, None, None]
        torch.testing.assert_close(v.grad, expected_gradient, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    unittest.main()
