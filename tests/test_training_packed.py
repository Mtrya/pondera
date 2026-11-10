"""Trainer smoke: loss decreases, resume replays the same loss sequence."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
from test_release import CASES, build_run

from pondera.chess.vocabulary import CHESS_VOCAB_SIZE
from pondera.data.merge_profile import build_profile
from pondera.data.packing import PackedDocuments, load_subsets, write_packed
from pondera.data.pipeline import produce_shards
from pondera.data.tokenize import tokenize_dataset
from pondera.models.vanilla import VanillaModel
from pondera.training.muon import newton_schulz5, split_parameters
from pondera.training.packed import TrainConfig, train

MODEL = dict(
    num_blocks=1,
    hidden_size=32,
    intermediate_size=48,
    num_heads=4,
    num_kv_heads=2,
)


def _ddp_worker(rank, store, packed, model_config, output, config):
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=2
    )
    try:
        torch.set_num_threads(1)
        output = Path(output)
        for name, updates, resume in (
            ("partial", 1, None),
            ("resumed", 2, output / "partial" / "checkpoint.pth"),
        ):
            torch.manual_seed(11)
            model = VanillaModel(**model_config)
            config.updates = updates
            config.checkpoint_every = updates
            train(
                model,
                model_config,
                packed,
                "50k",
                config,
                output / name,
                resume=resume,
                device="cpu",
            )
    finally:
        dist.destroy_process_group()


class MuonTests(unittest.TestCase):
    def test_newton_schulz_orthogonalizes(self):
        torch.manual_seed(0)
        matrix = torch.randn(8, 8)
        update = newton_schulz5(matrix)
        singular = torch.linalg.svdvals(update)
        # The quintic iteration compresses the singular spectrum toward one
        # without finishing it: five steps leave a bounded band around one,
        # far tighter than the input's spread (here [0.78, 5.49]).
        self.assertTrue(bool(((singular - 1).abs() < 0.5).all()))

    def test_split_rule(self):
        model = VanillaModel(**MODEL, vocab_size=CHESS_VOCAB_SIZE)
        muon, adamw = split_parameters(model)
        muon_names = {
            name for name, p in model.named_parameters() if any(p is m for m in muon)
        }
        self.assertTrue(all(p.ndim == 2 for p in muon))
        self.assertTrue(any(p is model.token_embedding.weight for p in adamw))
        self.assertTrue(any(p is model.head.weight for p in adamw))
        self.assertTrue(
            all(
                "token_embedding" not in name and "head" not in name
                for name in muon_names
            )
        )
        for block in model.blocks:
            self.assertTrue(any(p is block.attention.q.weight for p in muon))
            self.assertTrue(any(p is block.norm_attention.weight for p in adamw))


class TrainerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        config, roots, run, provider = build_run(
            self.root,
            cases=CASES,
            recipe_text="[recipe]\npresentations = {zero = 0.2, full = 0.5, random = 0.3}",
            seed=0,
        )
        next(root for root in roots if root["presentation"] == "zero")["split"] = "val"
        source = run / "release"
        produce_shards(
            roots, source, config, sidecar_for_root=provider, shard_size=2, workers=1
        )
        _, dictionaries = build_profile(
            [source],
            [CHESS_VOCAB_SIZE + 64],
            min_count=1,
            per_length_limit=64,
            block=4,
            max_rounds=64,
        )
        tokenized = self.root / "tokenized"
        tokenize_dataset(source, tokenized, dictionaries[CHESS_VOCAB_SIZE + 64])
        write_packed(tokenized, self.root / "packed", seed=640)
        self.model_config = dict(
            MODEL,
            vocab_size=CHESS_VOCAB_SIZE + len(dictionaries[CHESS_VOCAB_SIZE + 64]),
        )

    def make_model(self):
        torch.manual_seed(11)
        return VanillaModel(**self.model_config)

    def config(self, updates):
        documents = PackedDocuments(self.root / "packed")
        train_indices = load_subsets(self.root / "packed")["50k"]
        return TrainConfig(
            microbatch_tokens=65_536,
            update_tokens=2 * int(documents.lengths[train_indices].sum()),
            updates=updates,
            val_every=2,
            checkpoint_every=updates,
        )

    def test_loss_decreases(self):
        model = self.make_model()
        result = train(
            model,
            self.model_config,
            self.root / "packed",
            "50k",
            self.config(6),
            self.root / "train-out",
        )
        losses = [record["loss"] for record in result["history"]]
        self.assertLess(losses[-1], losses[0])
        self.assertIn("val_loss", result["history"][-1])
        streamed = [
            json.loads(line)
            for line in (self.root / "train-out" / "history.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertEqual(streamed, result["history"])

    def check_resume_replays_the_same_losses(self, device):
        reference = train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            "50k",
            self.config(4),
            self.root / "full",
            device=device,
        )
        partial = train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            "50k",
            self.config(2),
            self.root / "partial",
            device=device,
        )
        resumed = train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            "50k",
            self.config(4),
            self.root / "resumed",
            resume=self.root / "partial" / "checkpoint.pth",
            device=device,
        )
        expected = [record["loss"] for record in reference["history"]]
        actual = [record["loss"] for record in partial["history"] + resumed["history"]]
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=5)
        streamed = [
            json.loads(line)["loss"]
            for name in ("partial", "resumed")
            for line in (self.root / name / "history.jsonl").read_text().splitlines()
        ]
        self.assertEqual(streamed, actual)

    def test_resume_replays_the_same_losses(self):
        self.check_resume_replays_the_same_losses("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA resume requires a GPU")
    def test_cuda_resume_replays_the_same_losses(self):
        self.check_resume_replays_the_same_losses("cuda")

    def test_resume_uses_requested_learning_rates(self):
        train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            "50k",
            self.config(1),
            self.root / "lr-partial",
        )
        config = self.config(2)
        config.muon_lr = 0.01
        config.adamw_lr = 0.00015
        train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            "50k",
            config,
            self.root / "lr-resumed",
            resume=self.root / "lr-partial" / "checkpoint.pth",
        )
        state = torch.load(
            self.root / "lr-resumed" / "checkpoint.pth", weights_only=True
        )
        rates = [
            [group["lr"] for group in optimizer["param_groups"]]
            for optimizer in state["optimizer_state_dicts"]
        ]
        self.assertEqual(rates, [[0.01], [0.00015]])

    def test_distributed_matches_single_process_and_resume(self):
        config = self.config(2)
        config.val_every = 1
        reference = train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            "50k",
            config,
            self.root / "reference",
        )
        output = self.root / "distributed"
        torch.multiprocessing.spawn(
            _ddp_worker,
            args=(
                self.root / "store",
                self.root / "packed",
                self.model_config,
                output,
                config,
            ),
            nprocs=2,
            join=True,
        )
        actual = []
        for name in ("partial", "resumed"):
            actual.extend(json.loads((output / name / "history.json").read_text()))
        expected = reference["history"]
        self.assertEqual(len(actual), len(expected))
        for observed, wanted in zip(actual, expected):
            for field in ("update", "micro_step", "tokens", "labeled_tokens", "epoch"):
                self.assertEqual(observed[field], wanted[field])
            self.assertAlmostEqual(observed["loss"], wanted["loss"], places=4)
            self.assertAlmostEqual(observed["val_loss"], wanted["val_loss"], places=4)

        reference_state = torch.load(
            self.root / "reference" / "checkpoint.pth", weights_only=True
        )
        distributed_state = torch.load(
            output / "resumed" / "checkpoint.pth", weights_only=True
        )
        reference_weights = reference_state["model_state_dict"]
        distributed_weights = distributed_state["model_state_dict"]
        self.assertEqual(distributed_weights.keys(), reference_weights.keys())
        self.assertTrue(
            all(not key.startswith("module.") for key in distributed_weights)
        )
        for key, wanted in reference_weights.items():
            torch.testing.assert_close(
                distributed_weights[key], wanted, rtol=1e-4, atol=1e-5
            )


if __name__ == "__main__":
    unittest.main()
