"""Vanilla model invariants: causality, bounded FFN, packed/eager equivalence."""

import unittest

import torch

from pondera.models.linear import LinearVanillaModel
from pondera.models.vanilla import VanillaModel, situ_glu, varlen_attention

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
                logits = original(ids, cu)
                checkpointed_logits = recomputed(ids, cu)
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
        from pondera.models.vanilla import eager_varlen_attention

        fused = varlen_attention(q, k, v, cu)
        reference = eager_varlen_attention(q, k, v, cu)
        self.assertLessEqual((fused - reference).abs().max().item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
