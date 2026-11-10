"""Trainer smoke: loss decreases, resume replays the same loss sequence."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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
from pondera.models.linear import LinearVanillaModel
from pondera.models.vanilla import VanillaModel
from pondera.training.config import TrainConfig
from pondera.training.muon import Muon, newton_schulz5, split_parameters
from pondera.training.packed import train

MODEL = dict(
    num_blocks=1,
    hidden_size=32,
    intermediate_size=48,
    num_heads=4,
    num_kv_heads=2,
)


def _ddp_worker(rank, store, packed, model_config, output, config, runs, model_class):
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=2
    )
    try:
        torch.set_num_threads(1)
        output = Path(output)
        for name, updates, resume in runs:
            torch.manual_seed(11)
            model = model_class(**model_config)
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
            )
    finally:
        dist.destroy_process_group()


class MuonTests(unittest.TestCase):
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
        optimizer = Muon(parameters, lr=0.02, momentum=0.95)
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
        return VanillaModel(**self.model_config)

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

    def write_training_recipe(self, path, architecture="vanilla", updates=2):
        path.write_text(f"""
[model]
architecture = "{architecture}"
num_blocks = 1
hidden_size = 32
intermediate_size = 48
num_heads = 4
num_kv_heads = 2

[training]
seed = 11
muon_lr = 0.01
adamw_lr = 0.0007
weight_decay = 0.05
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
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )

    def test_toml_training_and_resume_match_uninterrupted_training(self):
        for architecture in ("vanilla", "linear"):
            with self.subTest(architecture=architecture):
                recipe = self.root / f"{architecture}.toml"
                full = self.root / f"{architecture}-full"
                partial = self.root / f"{architecture}-partial"
                self.write_training_recipe(recipe, architecture)
                result = self.run_training_command(recipe, full)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.write_training_recipe(recipe, architecture, updates=1)
                result = self.run_training_command(recipe, partial)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.write_training_recipe(recipe, architecture)
                result = self.run_training_command(
                    recipe, partial, partial / "checkpoint.pth"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                expected = torch.load(full / "checkpoint.pth", weights_only=True)
                actual = torch.load(partial / "checkpoint.pth", weights_only=True)
                self.assertEqual(actual["arch"], architecture)
                self.assertEqual(actual["model_config"], self.model_config)
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
                self.assertEqual(
                    [
                        optimizer["param_groups"][0]["lr"]
                        for optimizer in actual["optimizer_state_dicts"]
                    ],
                    [0.01, 0.0007],
                )
                self.assertTrue(
                    all(
                        optimizer["param_groups"][0]["weight_decay"] == 0.05
                        for optimizer in actual["optimizer_state_dicts"]
                    )
                )
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
                    [record["loss"] for record in reference_history],
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
                result = self.run_training_command(
                    recipe, partial, partial / "checkpoint.pth"
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(
                    [
                        json.loads(line)
                        for line in (partial / "history.jsonl").read_text().splitlines()
                    ],
                    resumed_history,
                )

    def test_invalid_recipe_does_not_start_training(self):
        recipe = self.root / "invalid.toml"
        self.write_training_recipe(recipe)
        original = recipe.read_text()
        for change in (
            original.replace("muon_lr = 0.01", "muon_lr = -0.01"),
            original.replace("num_heads = 4", "num_heads = 3"),
            original + '\ndevice = "cpu"\n',
            original.replace("documents = 3", "documents = true"),
            original.replace("checkpoint_keep_last = 2", "checkpoint_keep_last = 0"),
        ):
            with self.subTest(recipe=change):
                recipe.write_text(change)
                output = self.root / "invalid-run"
                result = self.run_training_command(recipe, output)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())

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
                LinearVanillaModel(**self.model_config),
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
        self.assertEqual(architectures, ["linear", "linear"])
        local_config = json.loads((output / "config.json").read_text())
        self.assertEqual(local_config["arch"], "linear")
        state = torch.load(output / "checkpoint.pth", weights_only=True)
        self.assertIsInstance(state["swanlab_run_id"], str)
        self.assertTrue(state["swanlab_run_id"])
        artifacts = list((output / "swanlog").glob("run-*/backup.swanlab"))
        self.assertTrue(artifacts)
        self.assertGreater(artifacts[0].stat().st_size, 0)

    def check_resume_replays_the_same_losses(self, device):
        reference = train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            self.config(4),
            self.root / "full",
            device=device,
            track=False,
        )
        partial = train(
            self.make_model(),
            self.model_config,
            self.root / "packed",
            self.config(2),
            self.root / "partial",
            device=device,
            track=False,
        )
        copied = self.root / "packed-copy"
        shutil.copytree(self.root / "packed", copied)
        resumed = train(
            self.make_model(),
            self.model_config,
            copied,
            self.config(4),
            self.root / "resumed",
            resume=self.root / "partial" / "checkpoint.pth",
            device=device,
            track=False,
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
        reference_state = torch.load(
            self.root / "full" / "checkpoint.pth", weights_only=True
        )
        resumed_state = torch.load(
            self.root / "resumed" / "checkpoint.pth", weights_only=True
        )
        for key, value in reference_state["model_state_dict"].items():
            torch.testing.assert_close(
                resumed_state["model_state_dict"][key], value, rtol=0, atol=0
            )

    def test_resume_replays_the_same_losses(self):
        self.check_resume_replays_the_same_losses("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA resume requires a GPU")
    def test_cuda_resume_replays_the_same_losses(self):
        self.check_resume_replays_the_same_losses("cuda")

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
        config.weight_decay = 0.7
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
        self.assertEqual(rates, [[0.01], [0.00015]])
        for optimizer in state["optimizer_state_dicts"]:
            self.assertEqual(optimizer["param_groups"][0]["weight_decay"], 0.7)

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
        for field in ("seed", "microbatch_tokens"):
            with self.subTest(field=field):
                config = self.config(2)
                setattr(config, field, getattr(config, field) + 1)
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
        with self.assertRaises(ValueError):
            train(
                LinearVanillaModel(**self.model_config),
                self.model_config,
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
                    ("partial", 1, None),
                    ("resumed", 2, output / "partial" / "checkpoint.pth"),
                ),
                VanillaModel,
            ),
            nprocs=2,
            join=True,
        )
        actual = []
        for name in ("partial", "resumed"):
            actual.extend(
                json.loads(line)
                for line in (output / name / "history.jsonl").read_text().splitlines()
            )
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
        for model_class in (VanillaModel, LinearVanillaModel):
            with self.subTest(architecture=model_class.architecture):
                root = self.root / model_class.architecture
                torch.manual_seed(11)
                reference = train(
                    model_class(**self.model_config),
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
                        model_class,
                    ),
                    nprocs=2,
                    join=True,
                )
                actual = torch.load(
                    output / "full" / "checkpoint.pth", weights_only=True
                )
                expected = torch.load(
                    root / "reference" / "checkpoint.pth", weights_only=True
                )
                self.assertEqual(actual["micro_step"], batches)
                self.assertEqual(actual["presented_documents"], len(indices))
                observed = [
                    json.loads(line)
                    for line in (output / "full" / "history.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(len(observed), reference["updates"])
                for record, wanted in zip(observed, reference["history"]):
                    for field in (
                        "micro_step",
                        "tokens",
                        "labeled_tokens",
                        "presentations",
                    ):
                        self.assertEqual(record[field], wanted[field])
                    self.assertAlmostEqual(record["loss"], wanted["loss"], places=4)
                    self.assertAlmostEqual(
                        record["val_loss"], wanted["val_loss"], places=4
                    )
                for key, weight in expected["model_state_dict"].items():
                    torch.testing.assert_close(
                        actual["model_state_dict"][key], weight, rtol=1e-4, atol=1e-5
                    )


if __name__ == "__main__":
    unittest.main()
