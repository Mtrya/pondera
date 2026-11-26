"""Trainer smoke: loss decreases, resume replays the same loss sequence."""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from datetime import timedelta
from pathlib import Path

import numpy as np
import swanlab
import torch
import torch.distributed as dist
from swanlab.toolkit.callback import SwanKitCallback
from test_release import CASES, build_run

from pondera.chess.merge import MergeDictionary
from pondera.chess.vocabulary import CHESS_VOCAB_SIZE
from pondera.data.dataset import load_manifest, write_manifest
from pondera.data.merge_profile import build_profile
from pondera.data.packing import (
    PackedDocuments,
    packed_batch_indices,
    select_documents,
)
from pondera.data.pipeline import produce_shards
from pondera.data.tokenize import encode_dataset
from pondera.evaluation.loss import evaluate_loss
from pondera.models import PonderaModel
from pondera.training.config import TrainConfig
from pondera.training.muon import Muon, newton_schulz5, split_parameters
from pondera.training.packed import train, validation_loss

MODEL = dict(
    segments=[{"blocks": ["softmax"]}],
    hidden_size=32,
    intermediate_size=48,
    num_heads=4,
    num_kv_heads=2,
)


def _ddp_worker(rank, store, packed, model_config, output, config, runs):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.set_num_threads(1)
        output = Path(output)
        for name, updates, resume in runs:
            torch.manual_seed(11)
            model = PonderaModel(**model_config)
            config.updates = updates
            config.checkpoint_every = 1
            train(
                model,
                model_config,
                packed,
                config,
                output / name,
                resume=resume,
                device="cpu",
                track=False,
                validation_internals=True,
                validation_document_steps=(1, 2),
            )
    finally:
        dist.destroy_process_group()


class MuonTests(unittest.TestCase):
    def test_hyperball_perpendicular_update(self):
        parameter = torch.nn.Parameter(torch.tensor([[3.0, 0.0], [0.0, 4.0]]))
        optimizer = Muon([parameter], lr=0.1, hyperball=True)
        parameter.grad = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        optimizer.step()
        # Equal singular values preserve this direction. The unit update is
        # perpendicular to W, so projection divides by sqrt(1 + 0.1**2).
        expected = torch.tensor([[2.98511157, -0.35179877], [0.35179877, 3.98014876]])
        torch.testing.assert_close(parameter, expected)

    def test_hyperball_preserves_initial_matrix_norms(self):
        torch.manual_seed(640)
        parameters = [
            torch.nn.Parameter(torch.randn(shape))
            for shape in ((8, 8), (8, 32), (32, 8))
        ]
        radii = [parameter.norm().detach().clone() for parameter in parameters]
        optimizer = Muon(parameters, lr=0.1, hyperball=True)
        for _ in range(4):
            for parameter in parameters:
                parameter.grad = torch.randn_like(parameter)
            optimizer.step()
            for parameter, radius in zip(parameters, radii, strict=True):
                torch.testing.assert_close(parameter.norm(), radius)

    def test_newton_schulz_orthogonalizes(self):
        torch.manual_seed(0)
        matrix = torch.randn(8, 8)
        update = newton_schulz5(matrix)
        singular = torch.linalg.svdvals(update.float())
        # The quintic iteration compresses the singular spectrum toward one
        # without finishing it: five steps leave a bounded band around one,
        # far tighter than the input's spread (here [0.78, 5.49]).
        self.assertTrue(bool(((singular - 1).abs() < 0.5).all()))

    def test_matches_pytorch_original_muon(self):
        torch.manual_seed(640)
        shapes = [(8, 8), (8, 32), (32, 8)]
        parameters = [torch.nn.Parameter(torch.randn(shape)) for shape in shapes]
        reference_parameters = [
            torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters
        ]
        optimizer = Muon(parameters, lr=0.02)
        reference = torch.optim.Muon(
            reference_parameters,
            lr=0.02,
            weight_decay=0,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            adjust_lr_fn="original",
        )
        for _ in range(2):
            for index, (parameter, expected) in enumerate(
                zip(parameters, reference_parameters)
            ):
                gradient = torch.randn_like(parameter) * (index + 1)
                parameter.grad = gradient.clone()
                expected.grad = gradient.clone()
            optimizer.step()
            reference.step()
        # BF16 addmm in the reference differs from Keller's separate matrix
        # operations by rounding; the two-step updates retain that tolerance.
        for parameter, expected in zip(parameters, reference_parameters):
            torch.testing.assert_close(parameter, expected, rtol=0.01, atol=0.002)

    def test_split_rule(self):
        model = PonderaModel(**MODEL, vocab_size=CHESS_VOCAB_SIZE)
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
        self.source = source
        encode_dataset(
            source, self.root / "packed", dictionaries[CHESS_VOCAB_SIZE + 64]
        )
        self.model_config = dict(
            MODEL,
            vocab_size=CHESS_VOCAB_SIZE + len(dictionaries[CHESS_VOCAB_SIZE + 64]),
        )

    def make_model(self):
        torch.manual_seed(11)
        return PonderaModel(**self.model_config)

    def config(self, updates):
        documents = PackedDocuments(self.root / "packed")
        train_indices = select_documents(
            PackedDocuments(self.root / "packed"), seed=640
        )
        return TrainConfig(
            microbatch_tokens=65_536,
            update_tokens=2 * int(documents.lengths[train_indices].sum()),
            updates=updates,
            val_every=2,
            checkpoint_every=updates,
        )

    def write_training_recipe(self, path, updates=2):
        path.write_text(f"""
[model]
hidden_size = 32
intermediate_size = 48
num_heads = 4
num_kv_heads = 2
block_attn_res = true
factorized_move_head = true

[[model.segments]]
blocks = [{{ attention = "kda", group_end = true }}]

[[model.segments]]
repeat = 2
blocks = ["kda", {{ attention = "softmax", navigation = true, group_end = true }}]

[[model.segments]]
blocks = [{{ attention = "softmax", value_source = "token_embedding", group_end = true }}]

[training]
seed = 11
optimizer = "hyperball"
muon_lr = 0.01
adamw_lr = 0.0007
warmup_fraction = 0.5
min_lr_ratio = 0.1
adamw_weight_decay = 0.05
microbatch_tokens = 1024
update_tokens = 1
updates = {updates}
documents = 3
max_document_tokens = 65536
val_every = 1
checkpoint_every = 1
checkpoint_keep_last = 2
""")

    def run_training_command(self, recipe, output, resume=None):
        command = [
            sys.executable,
            "-m",
            "pondera.training.packed",
            "--config",
            str(recipe),
            "--packed",
            str(self.root / "packed"),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--no-track",
        ]
        if resume is not None:
            command.extend(["--resume", str(resume)])
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )

    def test_init_seed_changes_weights_without_changing_document_selection(self):
        recipe = self.root / "model.toml"
        self.write_training_recipe(recipe, updates=1)
        recipe_text = recipe.read_text()
        states = {}
        for init_seed in (None, 11, 0):
            if init_seed is not None:
                recipe.write_text(f"{recipe_text}\ninit_seed = {init_seed}\n")
            output = self.root / f"init-{init_seed}"
            result = self.run_training_command(recipe, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            states[init_seed] = torch.load(output / "checkpoint.pth", weights_only=True)
            self.assertEqual(states[init_seed]["train_config"]["init_seed"], init_seed)

        self.assertEqual(
            states[None]["document_indices"], states[0]["document_indices"]
        )
        name = "token_embedding.weight"
        reference = states[None]["model_state_dict"][name]
        self.assertTrue(torch.equal(reference, states[11]["model_state_dict"][name]))
        self.assertFalse(torch.equal(reference, states[0]["model_state_dict"][name]))

    def test_toml_training_and_resume_match_uninterrupted_training(self):
        recipe = self.root / "model.toml"
        full = self.root / "full"
        partial = self.root / "partial"
        self.write_training_recipe(recipe)
        result = self.run_training_command(recipe, full)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_training_command(
            recipe, partial, full / "checkpoint-00000001.pth"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = torch.load(full / "checkpoint.pth", weights_only=True)
        actual = torch.load(partial / "checkpoint.pth", weights_only=True)
        self.assertEqual(actual["arch"], "pondera")
        settings = tomllib.loads(recipe.read_text())["model"]
        settings["vocab_size"] = self.model_config["vocab_size"]
        self.assertEqual(actual["model_config"], settings)
        self.assertEqual(actual["presented_documents"], 6)
        self.assertEqual(
            actual["selection"],
            {
                "documents": 3,
                "max_document_tokens": 65536,
                "validation_documents": 1,
            },
        )
        self.assertEqual(actual["train_config"]["seed"], 11)
        self.assertEqual(actual["train_config"]["microbatch_tokens"], 1024)
        self.assertEqual(actual["train_config"]["update_tokens"], 1)
        self.assertEqual(actual["train_config"]["checkpoint_keep_last"], 2)
        self.assertEqual(actual["train_config"]["warmup_fraction"], 0.5)
        self.assertEqual(actual["train_config"]["min_lr_ratio"], 0.1)
        for optimizer, expected_lr in zip(
            actual["optimizer_state_dicts"], (0.001, 0.00007)
        ):
            self.assertAlmostEqual(optimizer["param_groups"][0]["lr"], expected_lr)
        matrix, adamw = actual["optimizer_state_dicts"]
        self.assertTrue(matrix["param_groups"][0]["hyperball"])
        self.assertEqual(adamw["param_groups"][0]["weight_decay"], 0.05)
        self.assertEqual(
            json.loads((partial / "config.json").read_text())["train_config"],
            actual["train_config"],
        )
        reference_history = [
            json.loads(line)
            for line in (full / "history.jsonl").read_text().splitlines()
        ]
        resumed_history = [
            json.loads(line)
            for line in (partial / "history.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            [record["loss"] for record in resumed_history],
            [record["loss"] for record in reference_history[1:]],
        )
        for name, weight in expected["model_state_dict"].items():
            torch.testing.assert_close(
                actual["model_state_dict"][name], weight, rtol=0, atol=0
            )
        # The requested model in TOML must not be silently replaced
        # with checkpoint settings during resume.
        recipe.write_text(
            recipe.read_text().replace("hidden_size = 32", "hidden_size = 64")
        )
        result = self.run_training_command(recipe, partial, partial / "checkpoint.pth")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            [
                json.loads(line)
                for line in (partial / "history.jsonl").read_text().splitlines()
            ],
            resumed_history,
        )

    def test_checkpoint_retention_and_resume_from_an_older_update(self):
        config = self.config(5)
        config.checkpoint_every = 1
        config.checkpoint_keep_last = 2
        output = self.root / "retained"
        output.mkdir()
        unrelated = output / "manual.pth"
        train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            config,
            output,
            track=False,
        )
        unrelated.write_bytes(b"independent artifact")

        def snapshots():
            return sorted(
                (torch.load(path, weights_only=True)["update"], path)
                for path in output.glob("checkpoint-*.pth")
            )

        saved = snapshots()
        self.assertEqual([step for step, _ in saved], [4, 5])
        self.assertEqual(
            torch.load(output / "checkpoint.pth", weights_only=True)["update"], 5
        )
        # The latest path must share storage with the numbered checkpoint.
        self.assertTrue((output / "checkpoint.pth").samefile(saved[-1][1]))
        config.updates = 4
        train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            config,
            output,
            resume=saved[0][1],
            track=False,
        )
        self.assertEqual([step for step, _ in snapshots()], [4])
        self.assertEqual(
            torch.load(output / "checkpoint.pth", weights_only=True)["update"], 4
        )
        config.updates = 6
        train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            config,
            output,
            resume=output / "checkpoint.pth",
            track=False,
        )
        saved = snapshots()
        self.assertEqual([step for step, _ in saved], [5, 6])
        config.checkpoint_keep_last = 1
        train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            config,
            output,
            resume=output / "checkpoint.pth",
            track=False,
        )
        saved = snapshots()
        self.assertEqual([step for step, _ in saved], [6])
        self.assertTrue((output / "checkpoint.pth").samefile(saved[0][1]))
        self.assertEqual(unrelated.read_bytes(), b"independent artifact")

    def test_warmup_cosine_rates_follow_update_and_document_budgets(self):
        states = []
        for budget in ("updates", "presentations"):
            with self.subTest(budget=budget):
                config = self.config(4)
                config.muon_lr = 0.02
                config.warmup_fraction = 0.5
                config.min_lr_ratio = 0.1
                config.checkpoint_every = 1
                config.checkpoint_keep_last = 4
                expected = (
                    (0.01, 0.00015),
                    (0.02, 0.0003),
                    (0.011, 0.000165),
                    (0.002, 0.00003),
                )
                if budget == "presentations":
                    # Three two-presentation updates, then one short final update.
                    # Warmup ends at 4/7; update three reaches 6/7, a 2/3 cosine phase.
                    config.updates = None
                    config.presentations = 7
                    config.warmup_fraction = 4 / 7
                    expected = (*expected[:2], (0.0065, 0.0000975), expected[-1])
                output = self.root / f"schedule-{budget}"
                result = train(
                    self.make_model(),
                    self.model_config,
                    self.root / "packed",
                    config,
                    output,
                    track=False,
                )
                self.assertEqual(len(result["history"]), len(expected))
                for record, (muon_lr, adamw_lr) in zip(result["history"], expected):
                    self.assertAlmostEqual(record["muon_lr"], muon_lr)
                    self.assertAlmostEqual(record["adamw_lr"], adamw_lr)
                first = torch.load(
                    output / "checkpoint-00000001.pth", weights_only=True
                )
                final = torch.load(output / "checkpoint.pth", weights_only=True)
                for optimizer, expected_lr in zip(
                    final["optimizer_state_dicts"], expected[-1]
                ):
                    self.assertAlmostEqual(
                        optimizer["param_groups"][0]["lr"], expected_lr
                    )
                states.append((first, final))
        for key, weight in states[0][0]["model_state_dict"].items():
            torch.testing.assert_close(
                states[1][0]["model_state_dict"][key], weight, rtol=0, atol=0
            )

        # Equivalent effective rates must produce the same first optimizer update.
        config = self.config(1)
        config.muon_lr = 0.01
        config.adamw_lr = 0.00015
        config.warmup_fraction = 0
        config.min_lr_ratio = 1
        train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            config,
            self.root / "constant",
            track=False,
        )
        constant = torch.load(self.root / "constant/checkpoint.pth", weights_only=True)
        for key, weight in constant["model_state_dict"].items():
            torch.testing.assert_close(
                states[0][0]["model_state_dict"][key], weight, rtol=0, atol=0
            )

    def test_invalid_direct_schedule_fails_before_creating_outputs(self):
        for field, value in (
            ("warmup_fraction", 1),
            ("min_lr_ratio", -0.1),
            ("min_lr_ratio", float("nan")),
        ):
            with self.subTest(field=field, value=value):
                config = self.config(1)
                setattr(config, field, value)
                output = self.root / "invalid-schedule"
                with self.assertRaises(ValueError):
                    train(
                        self.make_model(),
                        self.model_config,
                        self.root / "packed",
                        config,
                        output,
                        track=False,
                    )
                self.assertFalse(output.exists())

    def test_absolute_presentation_cap_and_resume_accounting(self):
        config = self.config(8)
        config.max_presented_documents = 7
        output = self.root / "capped"
        first = train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            config,
            output,
            track=False,
        )
        self.assertEqual(first["presented_documents"], 7)
        self.assertEqual(
            json.loads((output / "progress.json").read_text())["presented_documents"], 7
        )
        config.max_presented_documents = 10
        second = train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            config,
            output,
            resume=output / "checkpoint.pth",
            track=False,
        )
        self.assertEqual(second["presented_documents"], 10)
        self.assertEqual(second["history"][-1]["presented_documents"], 10)
        config.max_presented_documents = 9
        with self.assertRaises(ValueError):
            train(
                self.make_model(),
                self.model_config,
                self.root / "packed",
                config,
                self.root / "exhausted",
                resume=output / "checkpoint.pth",
                track=False,
            )

    def test_validation_includes_documents_above_training_length_filter(self):
        documents = PackedDocuments(self.root / "packed")
        longest = int(np.argmax(documents.lengths))
        splits = np.memmap(self.root / "packed" / "split.u8", np.uint8, mode="r+")
        splits[longest] = 1
        splits.flush()
        documents = PackedDocuments(self.root / "packed")
        config = self.config(1)
        config.max_document_tokens = int(documents.lengths[documents.split == 0].min())
        config.val_every = 1
        output = self.root / "all-validation"
        result = train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            config,
            output,
            track=False,
        )
        state = torch.load(output / "checkpoint.pth", weights_only=True)
        self.assertEqual(
            state["selection"]["validation_documents"], int((splits == 1).sum())
        )
        # Every document supplies length minus one within-document next-token targets.
        expected_labels = int((documents.lengths[documents.split == 1] - 1).sum())
        self.assertEqual(result["history"][-1]["val_labeled_tokens"], expected_labels)
        model = self.make_model()
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        loss, _, _ = validation_loss(
            model, documents, 65536, "cpu", np.flatnonzero(splits == 1)
        )
        self.assertFalse(model.training)
        self.assertAlmostEqual(loss, result["history"][-1]["val_loss"], places=6)

    def test_checkpoint_loss_matches_uniform_distribution_and_training_evaluation(self):
        output = self.root / "loss-reference"
        train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            self.config(1),
            output,
            track=False,
        )
        checkpoint = output / "checkpoint.pth"
        model = self.make_model()
        state = torch.load(checkpoint, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        documents = PackedDocuments(self.root / "packed")
        expected, labels, _ = validation_loss(
            model, documents, 65536, "cpu", np.flatnonzero(documents.split == 1)
        )
        result = evaluate_loss(checkpoint, self.root / "packed", batch_tokens=32)
        self.assertAlmostEqual(result["loss"], expected, delta=1e-6)
        self.assertEqual(result["labeled_tokens"], labels)
        self.assertEqual(
            sum(row["tokens"] for row in result["document_losses"]), labels
        )
        for value in state["model_state_dict"].values():
            value.zero_()
        constant = self.root / "uniform.pth"
        torch.save(state, constant)
        # A zero-logit model assigns 1/vocabulary probability to every next token.
        result = evaluate_loss(constant, self.root / "packed")
        self.assertAlmostEqual(
            result["loss"], math.log(self.model_config["vocab_size"]), places=6
        )

    def test_distributed_checkpoint_loss_preserves_all_document_results(self):
        output = self.root / "distributed-loss"
        train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            self.config(1),
            output,
            track=False,
        )
        packed = self.root / "packed"
        split = np.memmap(packed / "split.u8", np.uint8, mode="r+")
        split[:] = 1
        split.flush()
        checkpoint = output / "checkpoint.pth"
        # Uneven batches and a single batch exercise sharding and a rank with no work.
        for budget in (32, 65536):
            with self.subTest(batch_tokens=budget):
                expected = evaluate_loss(checkpoint, packed, batch_tokens=budget)
                destination = self.root / f"loss-{budget}"
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc-per-node=2",
                        "-m",
                        "pondera.evaluation.loss",
                        "--checkpoint",
                        str(checkpoint),
                        "--packed",
                        str(packed),
                        "--output",
                        str(destination),
                        "--batch-tokens",
                        str(budget),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                actual = json.loads((destination / "summary.json").read_text())
                rows = [
                    json.loads(line)
                    for line in (destination / "documents.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(rows, expected["document_losses"])
                self.assertEqual(actual["loss"], expected["loss"])
                self.assertEqual(actual["labeled_tokens"], expected["labeled_tokens"])
                self.assertEqual(actual["documents"], len(split))
                self.assertEqual(actual["world_size"], 2)

    def test_loss_decreases(self):
        model = self.make_model()
        result = train(
            model,
            self.model_config,
            self.root / "packed",
            self.config(6),
            self.root / "train-out",
            track=False,
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

    def test_swanlab_offline_tracking(self):
        events = []
        architectures = []

        class Recorder(SwanKitCallback):
            def __str__(self):
                return "training-event-recorder"

            def on_log(self, data, step, *args, **kwargs):
                events.append((data.copy(), step))
                architectures.append(swanlab.config["arch"])

        config = self.config(2)
        previous_mode = os.environ.get("SWANLAB_MODE")
        os.environ["SWANLAB_MODE"] = "offline"
        output = self.root / "tracked"
        try:
            swanlab.register_callbacks([Recorder()])
            result = train(
                PonderaModel(**self.model_config),
                self.model_config,
                self.root / "packed",
                config,
                output,
                track=True,
            )
        finally:
            if previous_mode is None:
                os.environ.pop("SWANLAB_MODE", None)
            else:
                os.environ["SWANLAB_MODE"] = previous_mode
        self.assertEqual([step for _, step in events], [1, 2])
        self.assertEqual([data for data, _ in events], result["history"])
        self.assertEqual(architectures, ["pondera", "pondera"])
        local_config = json.loads((output / "config.json").read_text())
        self.assertEqual(local_config["arch"], "pondera")
        state = torch.load(output / "checkpoint.pth", weights_only=True)
        self.assertIsInstance(state["swanlab_run_id"], str)
        self.assertTrue(state["swanlab_run_id"])
        artifacts = list((output / "swanlog").glob("run-*/backup.swanlab"))
        self.assertTrue(artifacts)
        self.assertGreater(artifacts[0].stat().st_size, 0)

    def check_resume_replays_the_same_losses(self, device, hyperball=False):
        config = self.config(4)
        config.optimizer = "hyperball" if hyperball else "muon"
        config.warmup_fraction = 0.5
        config.checkpoint_every = 1
        config.checkpoint_keep_last = 4
        reference = train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            config,
            self.root / "full",
            device=device,
            track=False,
        )
        copied = self.root / "packed-copy"
        shutil.copytree(self.root / "packed", copied)
        reference_state = torch.load(
            self.root / "full" / "checkpoint.pth", weights_only=True
        )
        for start in (1, 3):
            with self.subTest(resume_update=start):
                output = self.root / f"resumed-{start}"
                resumed = train(
                    self.make_model(),
                    self.model_config,
                    copied,
                    config,
                    output,
                    resume=self.root / "full" / f"checkpoint-{start:08d}.pth",
                    device=device,
                    track=False,
                )
                for actual, expected in zip(
                    resumed["history"], reference["history"][start:], strict=True
                ):
                    for field in ("loss", "muon_lr", "adamw_lr"):
                        self.assertAlmostEqual(actual[field], expected[field], places=5)
                streamed = [
                    json.loads(line)
                    for line in (output / "history.jsonl").read_text().splitlines()
                ]
                self.assertEqual(streamed, resumed["history"])
                resumed_state = torch.load(output / "checkpoint.pth", weights_only=True)
                for key, value in reference_state["model_state_dict"].items():
                    torch.testing.assert_close(
                        resumed_state["model_state_dict"][key], value, rtol=0, atol=0
                    )

    def test_resume_replays_the_same_losses(self):
        self.check_resume_replays_the_same_losses("cpu")

    def test_hyperball_resume_replays_the_same_losses(self):
        self.check_resume_replays_the_same_losses("cpu", hyperball=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA resume requires a GPU")
    def test_cuda_resume_replays_the_same_losses(self):
        # Exact weight replay requires deterministic FlashAttention backward.
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        try:
            self.check_resume_replays_the_same_losses("cuda")
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            if workspace is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace

    def test_resume_uses_requested_optimizer_settings(self):
        train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            self.config(1),
            self.root / "lr-partial",
            track=False,
        )
        config = self.config(2)
        config.muon_lr = 0.01
        config.adamw_lr = 0.00015
        config.min_lr_ratio = 0.25
        config.adamw_weight_decay = 0.7
        train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            config,
            self.root / "lr-resumed",
            resume=self.root / "lr-partial" / "checkpoint.pth",
            track=False,
        )
        state = torch.load(
            self.root / "lr-resumed" / "checkpoint.pth", weights_only=True
        )
        rates = [
            [group["lr"] for group in optimizer["param_groups"]]
            for optimizer in state["optimizer_state_dicts"]
        ]
        self.assertEqual(rates, [[0.0025], [0.0000375]])
        self.assertEqual(
            state["optimizer_state_dicts"][1]["param_groups"][0]["weight_decay"], 0.7
        )

    def test_resume_rejects_changed_sampler_and_architecture(self):
        output = self.root / "sampler-partial"
        train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            self.config(1),
            output,
            track=False,
        )
        checkpoint = output / "checkpoint.pth"
        for field, value in (
            ("seed", self.config(2).seed + 1),
            ("init_seed", self.config(2).seed + 1),
            ("microbatch_tokens", self.config(2).microbatch_tokens + 1),
            ("optimizer", "muon"),
        ):
            with self.subTest(field=field):
                config = self.config(2)
                setattr(config, field, value)
                with self.assertRaises(ValueError):
                    train(
                        self.make_model(),
                        self.model_config,
                        self.root / "packed",
                        config,
                        self.root / f"changed-{field}",
                        resume=checkpoint,
                        track=False,
                    )
        config = self.config(2)
        config.documents = 1
        with self.assertRaises(ValueError):
            train(
                self.make_model(),
                self.model_config,
                self.root / "packed",
                config,
                self.root / "changed-selection",
                resume=checkpoint,
                track=False,
            )
        changed = {**self.model_config, "segments": [{"blocks": ["kda"]}]}
        with self.assertRaises(ValueError):
            train(
                PonderaModel(**changed),
                changed,
                self.root / "packed",
                self.config(2),
                self.root / "changed-architecture",
                resume=checkpoint,
                track=False,
            )

    def test_resume_rejects_new_dataset_and_reordered_dictionary(self):
        output = self.root / "identity-partial"
        train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            self.config(1),
            output,
            track=False,
        )
        manifest = load_manifest(self.root / "packed" / "packing.json")
        for name, entries in (
            ("new-dataset", manifest["merge_entries"]),
            ("new-dictionary", manifest["merge_entries"][::-1]),
        ):
            with self.subTest(name=name):
                packed = self.root / f"{name}-packed"
                encode_dataset(self.source, packed, MergeDictionary(entries))
                if name == "new-dataset":
                    changed = load_manifest(packed / "packing.json")
                    changed["source_dataset_id"] = "another-dataset"
                    write_manifest(packed / "packing.json", changed)
                self.assertEqual(
                    PackedDocuments(packed).manifest["vocabulary"],
                    self.model_config["vocab_size"],
                )
                with self.assertRaises(ValueError):
                    train(
                        self.make_model(),
                        self.model_config,
                        packed,
                        self.config(2),
                        self.root / f"{name}-out",
                        resume=output / "checkpoint.pth",
                        track=False,
                    )

    def test_history_follows_checkpoint_on_same_directory_resume(self):
        output = self.root / "history-resume"
        train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            self.config(1),
            output,
            track=False,
        )
        old_checkpoint = self.root / "old-checkpoint.pth"
        shutil.copy2(output / "checkpoint.pth", old_checkpoint)
        train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            self.config(3),
            output,
            resume=old_checkpoint,
            track=False,
        )
        history_path = output / "history.jsonl"
        expected = [
            json.loads(line)["loss"] for line in history_path.read_text().splitlines()
        ]
        train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            self.config(3),
            output,
            resume=old_checkpoint,
            track=False,
        )
        records = [json.loads(line) for line in history_path.read_text().splitlines()]
        self.assertEqual([record["update"] for record in records], [1, 2, 3])
        self.assertEqual([record["loss"] for record in records], expected)
        with history_path.open("a") as handle:
            handle.write('{"update":')
        train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            self.config(4),
            output,
            resume=output / "checkpoint.pth",
            track=False,
        )
        records = [json.loads(line) for line in history_path.read_text().splitlines()]
        self.assertEqual([record["update"] for record in records], [1, 2, 3, 4])
        self.assertEqual([record["loss"] for record in records[:3]], expected)

    def test_fractional_presentations_and_resume(self):
        documents = PackedDocuments(self.root / "packed")
        count = len(select_documents(PackedDocuments(self.root / "packed"), seed=640))

        def config(presentations):
            result = self.config(1)
            result.updates = None
            result.presentations = presentations
            result.microbatch_tokens = int(documents.lengths.max())
            result.update_tokens = 1
            result.val_every = 0
            return result

        reference = train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            config(1.5),
            self.root / "presentation-full",
            track=False,
        )
        partial = train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            config(0.75),
            self.root / "presentation-partial",
            track=False,
        )
        resumed = train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            config(1.5),
            self.root / "presentation-resumed",
            resume=self.root / "presentation-partial" / "checkpoint.pth",
            track=False,
        )
        expected_count = count + count // 2
        self.assertEqual(reference["presented_documents"], expected_count)
        self.assertEqual(reference["presentations"], expected_count / count)
        # Extending a completed target preserves document progress. The old
        # partial optimizer update need not match an originally longer run.
        self.assertEqual(partial["presented_documents"], int(count * 0.75))
        self.assertEqual(resumed["presented_documents"], expected_count)
        expected_new_tokens = sum(
            int(documents.lengths[batch].sum())
            for batch, _ in packed_batch_indices(
                documents,
                select_documents(documents, seed=640),
                640,
                config(1.5).microbatch_tokens,
                start_document=partial["presented_documents"],
                presentations=1.5,
            )
        )
        self.assertEqual(
            sum(record["tokens"] for record in resumed["history"]), expected_new_tokens
        )

    def test_update_stop_keeps_schedule_and_paired_validation(self):
        documents = PackedDocuments(self.root / "packed")
        config = self.config(1)
        config.updates = None
        config.update_tokens = 1
        config.microbatch_tokens = int(documents.lengths.max())
        output = self.root / "stages"
        reference = train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            config,
            self.root / "reference",
            track=False,
        )
        first = train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            config,
            output,
            track=False,
            stop_after_update=1,
            validation_internals=True,
            validation_document_steps=(1,),
        )
        self.assertEqual(first["updates"], 1)
        self.assertEqual(
            first["history"][0]["muon_lr"], reference["history"][0]["muon_lr"]
        )
        probe = first["history"][0]["internals"]
        self.assertEqual(
            probe["documents"], [int(np.flatnonzero(documents.split == 1)[0])]
        )

        checkpoint = output / "checkpoint.pth"
        independently_scored = evaluate_loss(checkpoint, self.root / "packed")
        evidence = output / "validation-00000001"
        scores = [
            json.loads(line)
            for line in (evidence / "documents.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(scores), independently_scored["documents"])
        for score, wanted in zip(
            scores, independently_scored["document_losses"], strict=True
        ):
            self.assertEqual(score["document"], wanted["document"])
            self.assertEqual(score["tokens"], wanted["tokens"])
            self.assertAlmostEqual(
                score["nll_sum"] / score["tokens"],
                wanted["nll_sum"] / wanted["tokens"],
                places=6,
            )

        resumed = train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            config,
            output,
            track=False,
            resume=checkpoint,
        )
        self.assertEqual(resumed["updates"], reference["updates"])
        for actual, expected in zip(
            resumed["history"], reference["history"][1:], strict=True
        ):
            for field in (
                "loss",
                "tokens",
                "presented_documents",
                "muon_lr",
                "adamw_lr",
            ):
                self.assertEqual(actual[field], expected[field])
        self.assertTrue((evidence / "summary.json").is_file())

    def test_distributed_matches_single_process_and_resume(self):
        config = self.config(2)
        config.update_tokens *= 2
        config.val_every = 1
        reference = train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            config,
            self.root / "reference",
            track=False,
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
                (
                    ("partial", 2, None),
                    ("resumed", 2, output / "partial" / "checkpoint-00000001.pth"),
                ),
            ),
            nprocs=2,
            join=True,
        )
        actual = []
        for name in ("partial", "resumed"):
            rows = [
                json.loads(line)
                for line in (output / name / "history.jsonl").read_text().splitlines()
            ]
            actual.extend(rows[:1] if name == "partial" else rows)
        expected = reference["history"]
        self.assertEqual(len(actual), len(expected))
        for observed, wanted in zip(actual, expected):
            for field in (
                "update",
                "micro_step",
                "tokens",
                "labeled_tokens",
                "val_labeled_tokens",
                "epoch",
            ):
                self.assertEqual(observed[field], wanted[field])
            self.assertAlmostEqual(observed["loss"], wanted["loss"], places=4)
            self.assertAlmostEqual(observed["val_loss"], wanted["val_loss"], places=4)
            self.assertEqual(observed["muon_lr"], wanted["muon_lr"])
            self.assertEqual(observed["adamw_lr"], wanted["adamw_lr"])

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

    def test_distributed_finishes_uneven_presentation(self):
        documents = PackedDocuments(self.root / "packed")
        indices = select_documents(PackedDocuments(self.root / "packed"), seed=640)
        config = self.config(1)
        config.updates = None
        config.microbatch_tokens = int(documents.lengths.max())
        config.val_every = 1
        batches = len(
            list(
                packed_batch_indices(
                    documents,
                    indices,
                    config.seed,
                    config.microbatch_tokens,
                    presentations=1.0,
                )
            )
        )
        self.assertEqual(batches % 2, 1)
        root = self.root / "uneven"
        root.mkdir()
        torch.manual_seed(11)
        reference = train(
            PonderaModel(**self.model_config),
            self.model_config,
            self.root / "packed",
            config,
            root / "reference",
            track=False,
        )
        output = root / "distributed"
        torch.multiprocessing.spawn(
            _ddp_worker,
            args=(
                root / "store",
                self.root / "packed",
                self.model_config,
                output,
                config,
                (("full", None, None),),
            ),
            nprocs=2,
            join=True,
        )
        actual = torch.load(output / "full" / "checkpoint.pth", weights_only=True)
        expected = torch.load(root / "reference" / "checkpoint.pth", weights_only=True)
        self.assertEqual(actual["micro_step"], batches)
        self.assertEqual(actual["presented_documents"], len(indices))
        observed = [
            json.loads(line)
            for line in (output / "full" / "history.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(observed), reference["updates"])
        for record, wanted in zip(observed, reference["history"]):
            for field in (
                "micro_step",
                "tokens",
                "labeled_tokens",
                "val_labeled_tokens",
                "presentations",
            ):
                self.assertEqual(record[field], wanted[field])
            self.assertAlmostEqual(record["loss"], wanted["loss"], places=4)
            self.assertAlmostEqual(record["val_loss"], wanted["val_loss"], places=4)
        # At the first update, momentum is proportional to the gradient.
        # Compare before Muon's BF16 orthogonalization can amplify tiny
        # differences from the distributed reduction order.
        for observed_optimizer, expected_optimizer in zip(
            actual["optimizer_state_dicts"],
            expected["optimizer_state_dicts"],
            strict=True,
        ):
            for parameter, state in expected_optimizer["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(
                        observed_optimizer["state"][parameter][name],
                        value,
                        rtol=1e-5,
                        atol=1e-8,
                    )


if __name__ == "__main__":
    unittest.main()
