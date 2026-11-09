"""Trainer smoke: loss decreases, resume replays the same loss sequence."""

import json
import tempfile
import unittest
from pathlib import Path

import chess
import torch
import torch.distributed as dist
from test_release import CASES, build_run

from pondera.chess.codec import decode_board, legal_move_indices
from pondera.chess.vocabulary import CHESS_VOCAB_SIZE, FINAL, MOVE
from pondera.data.merge_profile import build_profile
from pondera.data.packing import PackedDocuments, load_subsets, write_packed
from pondera.data.pipeline import produce_shards
from pondera.data.tokenize import tokenize_dataset
from pondera.models.vanilla import VanillaModel
from pondera.training.muon import newton_schulz5, split_parameters
from pondera.training.packed import TrainConfig, train, zero_move_accuracy

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

    def test_zero_move_accuracy_masks_illegal_tokens(self):
        documents = PackedDocuments(self.root / "packed")
        index = next(
            i
            for i, (split, presentation) in enumerate(
                zip(documents.split, documents.presentation)
            )
            if split == 1 and presentation == 0
        )
        document = documents.document(index)
        final = next(i for i, token in enumerate(document) if token == FINAL)
        fen, _ = decode_board(document[:final])
        target = int(document[final + 1])
        legal = [MOVE + move for move in legal_move_indices(chess.Board(fen))]
        illegal = next(
            token for token in range(MOVE, CHESS_VOCAB_SIZE) if token not in legal
        )
        model = VanillaModel(**dict(self.model_config, num_blocks=0))
        with torch.no_grad():
            model.token_embedding.weight.fill_(1)
            model.head.weight.zero_()
            model.head.weight[target].fill_(1 / self.model_config["hidden_size"])
            model.head.weight[illegal].fill_(2 / self.model_config["hidden_size"])
        result = zero_move_accuracy(model, documents, 65_536, "cpu")
        self.assertEqual(result, {"correct": 1, "count": 1, "accuracy": 1.0})

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
        self.assertIn("val_move_accuracy", result["history"][-1])
        self.assertEqual(result["history"][-1]["val_zero_documents"], 1)

    def test_resume_replays_the_same_losses(self):
        full = self.make_model()
        reference = train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            "50k",
            self.config(4),
            self.root / "full",
        )
        partial = train(
            full,
            self.model_config,
            self.root / "packed",
            "50k",
            self.config(2),
            self.root / "partial",
        )
        resumed = train(
            full,
            self.model_config,
            self.root / "packed",
            "50k",
            self.config(4),
            self.root / "resumed",
            resume=self.root / "partial" / "checkpoint.pth",
        )
        expected = [record["loss"] for record in reference["history"]]
        actual = [record["loss"] for record in partial["history"] + resumed["history"]]
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=5)

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
            self.assertAlmostEqual(
                observed["val_move_accuracy"], wanted["val_move_accuracy"], places=4
            )
            self.assertEqual(
                observed["val_zero_documents"], wanted["val_zero_documents"]
            )

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
