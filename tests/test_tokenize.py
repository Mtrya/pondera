"""Real teacher traces qualify authoritative text and merged training inputs."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from test_release import CASES, build_run
from tools.export_hf import export_hf

from pondera.chess.merge import MergeDictionary, decode
from pondera.chess.vocabulary import CHESS_VOCAB_SIZE, FINAL
from pondera.data.config import load_config
from pondera.data.dataset import (
    HF_COLUMNS,
    TracePositions,
    load_manifest,
    parquet_sources,
    trace_data_identity,
)
from pondera.data.merge_profile import build_profile, load_full_rows, write_artifacts
from pondera.data.pipeline import produce_shards
from pondera.data.text import ROLE_MIXED, trace_atoms
from pondera.data.tokenize import tokenize_dataset
from pondera.data.trace import ROLE_FINAL_MOVE, load_sidecar, synthesize_root
from pondera.models.checkpoint import load_model, read_checkpoint, save_checkpoint
from pondera.models.policy import create_model
from pondera.training.trace import TraceCollator, TrainConfig, trace_loss_parts, train

FIXTURES = Path(__file__).parent / "fixtures" / "trace_v2"
MODEL = {"num_blocks": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2}


class TextTests(unittest.TestCase):
    def test_text_recovers_producer_atoms_for_every_recorded_case(self):
        config = load_config()
        for fixture in sorted(FIXTURES.iterdir()):
            root = json.loads((fixture / "root.json").read_text())
            result = synthesize_root(
                load_sidecar(fixture / "sidecar.jsonl"), root, config
            )
            for row in result.rows.values():
                with self.subTest(case=fixture.name, presentation=row.presentation):
                    np.testing.assert_array_equal(
                        trace_atoms(row.trace_text), row.tokens
                    )


class TokenizeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        config, roots, run, provider = build_run(
            self.root,
            cases=CASES,
            recipe_text="[recipe]\npresentations = {full = 1.0}",
        )
        roots[0]["split"] = "val"
        self.source = run / "release"
        produce_shards(
            roots,
            self.source,
            config,
            sidecar_for_root=provider,
            shard_size=2,
            workers=1,
        )
        self.original = self.rows(self.source)
        entries = []
        for row in self.original:
            atoms = row["trace_token_ids"]
            final = atoms.index(FINAL)
            entries.append(tuple(atoms[final : final + 8]))
            entries.append(tuple(atoms[:2]))
            entries.append(tuple(atoms[-2:]))
        self.dictionary = MergeDictionary(tuple(dict.fromkeys(entries)))

    @staticmethod
    def rows(directory):
        return [
            row
            for path, _, _ in parquet_sources(directory)
            for row in pq.read_table(path).to_pylist()
        ]

    def assert_preserved(self, actual):
        self.assertEqual(len(actual), len(self.original))
        for expected, result in zip(self.original, actual):
            self.assertEqual(
                {
                    name: value
                    for name, value in expected.items()
                    if name not in {"trace_token_ids", "trace_len"}
                },
                {
                    name: value
                    for name, value in result.items()
                    if name not in {"trace_token_ids", "trace_len"}
                },
            )
            self.assertEqual(result["trace_len"], len(result["trace_token_ids"]))
            np.testing.assert_array_equal(
                decode(result["trace_token_ids"], self.dictionary),
                expected["trace_token_ids"],
            )

    def test_configured_workflow_fits_one_budget_and_encodes_all_rows(self):
        for budget in (0, 8):
            with self.subTest(budget=budget):
                config = self.root / "data.toml"
                config.write_text(
                    Path("configs/data.toml")
                    .read_text()
                    .replace("budget = 16384", f"budget = {budget}")
                    .replace("min_count = 8", "min_count = 1")
                    .replace("# max_roots = 128", "max_roots = 3")
                    .replace("batch_size = 128", "batch_size = 2")
                )
                output = self.root / f"training-{budget}"
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pondera.data",
                        "encode",
                        "--config",
                        str(config),
                        "--input",
                        str(self.source),
                        "--output",
                        str(output),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.dictionary = MergeDictionary.read(output / "merged_vocabulary.txt")
                self.assertLessEqual(len(self.dictionary), budget)
                if budget:
                    self.assertGreater(len(self.dictionary), 0)
                self.assert_preserved(self.rows(output))
                document = json.loads((output / "profile" / "curve.json").read_text())
                self.assertEqual(document["settings"]["budget"], budget)
                self.assertEqual(document["corpus"]["loaded"]["rows"], 3)
                self.assertEqual(len(document["tiers"]), 1)
                self.assertGreater(document["timing"]["workflow_peak_rss_bytes"], 0)
                self.assertEqual(len(TracePositions(output, "train")), 4)

    def test_conversion_replaces_the_cache_from_authoritative_text(self):
        path = self.source / "shard-000000" / "data.parquet"
        table = pq.read_table(path)
        corrupt = [[CHESS_VOCAB_SIZE + 100]] * len(table)
        table = table.set_column(
            table.schema.get_field_index("trace_token_ids"),
            "trace_token_ids",
            pa.array(corrupt, type=pa.list_(pa.int32())),
        )
        pq.write_table(table, path)
        output = self.root / "encoded"
        result = tokenize_dataset(self.source, output, self.dictionary, batch_size=1)
        self.assertEqual(result["rows"], len(self.original))
        self.assertLess(result["tokens"], result["base_atoms"])
        self.assert_preserved(self.rows(output))
        first = pq.ParquetFile(output / "shard-000000" / "data.parquet")
        self.assertEqual(
            first.schema_arrow.field("trace_token_ids").type, pa.list_(pa.int32())
        )
        self.assertEqual(first.schema_arrow.names, table.schema.names)
        again = self.root / "encoded-again"
        tokenize_dataset(output, again, self.dictionary, batch_size=3)
        self.assertEqual(self.rows(again), self.rows(output))

    def test_hf_input_creates_ids_and_preserves_all_shared_fields(self):
        exported = self.root / "hf"
        export_hf(self.source, exported, batch_size=2)
        output = self.root / "hf-encoded"
        tokenize_dataset(exported, output, self.dictionary, batch_size=1)
        expected = [row for row in self.original if row["split"] == "train"]
        expected += [row for row in self.original if row["split"] == "val"]
        actual = self.rows(output)
        self.assertEqual(
            [{name: row[name] for name in HF_COLUMNS} for row in actual],
            [{name: row[name] for name in HF_COLUMNS} for row in expected],
        )
        self.assertEqual(
            set(actual[0]),
            {*HF_COLUMNS, "trace_token_ids", "trace_len"},
        )
        self.assertEqual(len(TracePositions(output, "train")), 4)
        self.assertEqual(len(TracePositions(output, "val")), 1)

    def test_merged_tokens_feed_real_loss_and_reload_with_the_dictionary(self):
        output = self.root / "encoded"
        tokenize_dataset(self.source, output, self.dictionary)
        data = TracePositions(output, "train")
        model = create_model({**MODEL, "merge_entries": data.dictionary.entries})
        batch = TraceCollator(data.dictionary)(
            [data[index] for index in range(len(data))]
        )
        self.assertTrue(bool((batch["trace_ids"] >= CHESS_VOCAB_SIZE).any()))
        self.assertTrue(bool((batch["trace_roles"] == ROLE_MIXED).any()))
        final = batch["final_move_index"]
        for index, position in enumerate(final):
            self.assertEqual(
                int(batch["trace_roles"][index, position]), ROLE_FINAL_MOVE
            )
            target = int(batch["trace_ids"][index, position])
            self.assertGreaterEqual(target, CHESS_VOCAB_SIZE)
            self.assertIn(FINAL, self.dictionary.expansion(target))
        loss, ce, logits, generated = trace_loss_parts(model, batch)
        self.assertEqual(
            int(generated.sum()), sum(data[index][2] for index in range(len(data)))
        )
        self.assertEqual(logits.shape[-1], CHESS_VOCAB_SIZE + len(self.dictionary))
        torch.testing.assert_close(loss, ce[generated].mean())
        training_batch = TraceCollator(data.dictionary, diagnostics=False)(
            [data[index] for index in range(len(data))]
        )
        self.assertEqual(set(training_batch), {"ids", "trace_ids"})
        training_loss, _, _, _ = trace_loss_parts(model, training_batch)
        torch.testing.assert_close(training_loss, loss)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(model.head.weight.grad).all()))
        checkpoint = self.root / "model.pth"
        model.eval()
        with torch.no_grad():
            expected = model.trace_logits(batch["ids"], batch["trace_ids"])
        save_checkpoint(checkpoint, model, MODEL)
        loaded = load_model(checkpoint)
        self.assertEqual(loaded.merge_dictionary, self.dictionary)
        with torch.no_grad():
            actual = loaded.trace_logits(batch["ids"], batch["trace_ids"])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        identity = trace_data_identity(output)
        MergeDictionary(tuple(reversed(self.dictionary.entries))).write(
            output / "merged_vocabulary.txt"
        )
        self.assertNotEqual(trace_data_identity(output), identity)

    def test_training_resume_uses_the_dataset_dictionary(self):
        data = self.root / "encoded"
        tokenize_dataset(self.source, data, self.dictionary)
        config = TrainConfig(
            model=MODEL,
            batch_size=2,
            epochs=1,
            workers=0,
            precision="fp32",
            train_probe_positions=0,
            val_every=100,
            save_every=100,
        )
        output = self.root / "train"
        first = train(config, data, output, max_steps=1)
        checkpoint = read_checkpoint(first)
        self.assertEqual(
            checkpoint["model_config"]["merge_entries"], self.dictionary.entries
        )
        final = train(config, data, output, resume=first, max_steps=2)
        self.assertEqual(read_checkpoint(final)["step"], 2)
        MergeDictionary(tuple(reversed(self.dictionary.entries))).write(
            data / "merged_vocabulary.txt"
        )
        with self.assertRaises(ValueError):
            train(config, data, output, resume=final, max_steps=2)

    def test_profile_preserves_splits_and_optional_empty_holdout(self):
        document, dictionaries = build_profile(
            [self.source],
            [0, 8],
            min_count=1,
            per_length_limit=32,
            block=4,
            max_rounds=64,
        )
        self.assertEqual(
            document["corpus"]["fitting"]["release_splits"], {"train": 4, "val": 1}
        )
        self.assertEqual(document["corpus"]["heldout"]["rows"], 0)
        self.assertEqual(document["corpus"]["selection"]["rows"], 5)
        self.assertIsNone(document["tiers"][1]["heldout"]["ratio"])
        self.assertEqual(document["tiers"][0]["measurement"]["ratio"], 1.0)
        measured = document["tiers"][1]["measurement"]
        self.assertEqual(measured["rows"], 5)
        self.assertEqual(
            sum(part["rows"] for part in measured["slices"]["release_split"].values()),
            5,
        )
        output = self.root / "profile"
        write_artifacts(output, document, dictionaries)
        self.assertEqual(
            MergeDictionary.read(output / "dictionary-8.tsv"), dictionaries[8]
        )
        self.assertEqual(
            json.loads((output / "curve.json").read_text())["corpus"]["loaded"]["rows"],
            5,
        )
        heldout, _ = build_profile(
            [self.source],
            [8],
            fit_split="train",
            min_count=1,
            per_length_limit=32,
            selection_stride=1,
            block=4,
            max_rounds=64,
        )
        self.assertEqual(heldout["corpus"]["fitting"]["rows"], 4)
        self.assertEqual(heldout["corpus"]["heldout"]["release_splits"], {"val": 1})
        again, repeats = build_profile(
            [self.source],
            [8],
            min_count=1,
            per_length_limit=32,
            selection_stride=1,
            block=4,
            max_rounds=64,
        )
        self.assertEqual(repeats[8], dictionaries[8])
        self.assertEqual(again["tiers"][0]["measurement"], measured)

    def test_profiler_expands_the_declared_cache_before_counting(self):
        output = self.root / "encoded"
        tokenize_dataset(self.source, output, self.dictionary)
        original = load_full_rows(self.source)
        merged = load_full_rows(output)
        for before, after in zip(original, merged):
            self.assertEqual(before.root_id, after.root_id)
            self.assertEqual(before.split, after.split)
            np.testing.assert_array_equal(before.atoms, after.atoms)

    def test_input_rejects_mismatched_trace_versions(self):
        path = self.source / "release.json"
        manifest = load_manifest(path)
        manifest["grammar_version"] = 1
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            load_full_rows(self.source)
        with self.assertRaises(ValueError):
            tokenize_dataset(self.source, self.root / "invalid", self.dictionary)


if __name__ == "__main__":
    unittest.main()
