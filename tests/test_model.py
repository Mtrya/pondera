"""Model invariants for causality, shared execution, and packed/cached parity."""

import math
import unittest
from importlib.util import find_spec

import torch

from pondera.models import PonderaModel
from pondera.models.internals import Internals, attach, detach
from pondera.models.softmax import eager_varlen_attention, situ_glu, varlen_attention

MODEL = dict(
    segments=[
        {"blocks": [{"attention": "kda", "group_end": True}]},
        {
            "repeat": 2,
            "blocks": [
                "kda",
                {"attention": "softmax", "group_end": True},
                {"attention": "kda", "group_end": True},
            ],
        },
        {"blocks": [{"attention": "softmax", "value_source": "token_embedding"}]},
    ],
    block_attn_res=True,
    hidden_size=32,
    intermediate_size=48,
    num_heads=4,
    num_kv_heads=2,
    vocab_size=61,
)


def packed_ids():
    return torch.randint(0, 61, (11,)), torch.tensor([0, 5, 11], dtype=torch.int32)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = PonderaModel(**MODEL).eval()
        with torch.no_grad():
            for read in self.model.depth_reads:
                read.query.normal_(std=0.15)

    def test_grouped_and_layerwise_loops_share_weights_in_their_declared_order(self):
        grouped = self.model
        prelude, core, coda = MODEL["segments"]
        split = PonderaModel(
            **{
                **MODEL,
                "segments": [
                    prelude,
                    *({"repeat": 2, "blocks": [block]} for block in core["blocks"]),
                    coda,
                ],
            }
        ).eval()
        split.load_state_dict(grouped.state_dict())
        ids, cu = packed_ids()
        for model, expected in (
            (grouped, [0, 1, 2, 3, 1, 2, 3, 4]),
            (split, [0, 1, 1, 2, 2, 3, 3, 4]),
        ):
            seen = []

            def record(index):
                return lambda *_: seen.append(index)

            hooks = [
                block.norm_attention.register_forward_pre_hook(record(index))
                for index, block in enumerate(model.blocks)
            ]
            try:
                with torch.no_grad():
                    model(ids, cu)
            finally:
                for hook in hooks:
                    hook.remove()
            self.assertEqual(seen, expected)
            with torch.inference_mode():
                full = model(ids[:5], [0, 5])
                _, cache = model.forward_cached(ids[:4], [0, 4])
                logits, _ = model.forward_cached(ids[4:5], [0, 5], cache)
            torch.testing.assert_close(logits[-1], full[-1], rtol=1e-5, atol=2e-6)

    def test_uniform_depth_attention_preserves_the_one_pass_function(self):
        model = PonderaModel(**MODEL).eval()
        ids, cu = packed_ids()
        with torch.inference_mode():
            repeated = model(ids, cu)
            model.override_repetitions(1)
            torch.testing.assert_close(repeated, model(ids, cu), rtol=1e-5, atol=2e-6)

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
        torch.testing.assert_close(packed, separate, rtol=1e-5, atol=3e-6)

    def test_activation_checkpointing_preserves_logits_and_gradients(self):
        ids, cu = packed_ids()
        labels = torch.randint(0, MODEL["vocab_size"], ids.shape)
        original = PonderaModel(**MODEL).train()
        recomputed = PonderaModel(**MODEL).train()
        original.load_state_dict(self.model.state_dict())
        recomputed.load_state_dict(original.state_dict())
        recomputed.activation_checkpointing = True

        def forward_with_saved_elements(model):
            saved = []

            def pack(tensor):
                saved.append(tensor.numel())
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                logits = model(ids, cu)
            return logits, sum(saved)

        logits, original_saved = forward_with_saved_elements(original)
        checkpointed_logits, checkpointed_saved = forward_with_saved_elements(
            recomputed
        )
        self.assertLess(checkpointed_saved, original_saved)
        torch.testing.assert_close(logits, checkpointed_logits, rtol=0, atol=0)
        torch.nn.functional.cross_entropy(logits, labels).backward()
        torch.nn.functional.cross_entropy(checkpointed_logits, labels).backward()
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

    def test_internals_collection_does_not_change_model_outputs(self):
        ids, cu = packed_ids()
        with torch.no_grad():
            expected = self.model(ids, cu)
        internals = Internals()
        attach(self.model, internals)
        try:
            with torch.no_grad():
                observed = self.model(ids, cu)
        finally:
            detach(self.model)
        self.assertTrue(torch.equal(expected, observed))

    def test_internals_report_distinguishes_repeated_passes(self):
        ids, cu = packed_ids()
        internals = Internals()
        attach(self.model, internals)
        try:
            with torch.no_grad():
                self.model(ids, cu)
            report = internals.report()
        finally:
            detach(self.model)
        # The key names are the contract consumed by the training history and
        # the SwanLab monitor.
        for key in (
            "residual/embedding",
            "residual/seg1/entry",
            "residual/seg1/pass1/exit",
            "residual/seg1/pass2/exit/token_rms_p99",
            "delta/seg1/pass1/grp1",
            "read/pos2/pass1/effective",
            "read/pos2/pass2/max_prob",
            "kda/lay02/pass1/beta",
            "kda/lay02/pass2/retention",
            "kda/lay02/pass2/gate",
        ):
            self.assertIn(key, report)
            self.assertTrue(math.isfinite(report[key]))
        self.assertFalse(any(key.startswith("delta/seg1/pass2/") for key in report))
        # The reported embedding RMS must match a direct computation through
        # the embedding module itself, an independent source.
        direct = self.model.token_embedding(ids).double().square().mean().sqrt()
        self.assertAlmostEqual(
            report["residual/embedding"], float(direct.detach()), places=3
        )

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
        # A batched projection and single-row projections accumulate FP32 sums differently.
        torch.testing.assert_close(full, torch.cat(steps), rtol=2e-5, atol=3e-5)

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
