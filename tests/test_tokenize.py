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
    DATA_FORMAT_VERSION,
    HF_COLUMNS,
    HF_SCHEMA,
    TracePositions,
    load_manifest,
    parquet_sources,
    trace_data_identity,
    write_manifest,
)
from pondera.data.merge_profile import build_profile, load_full_rows, write_artifacts
from pondera.data.pipeline import EngineTimeoutError, produce_shards, validate_release
from pondera.data.text import ROLE_MIXED, trace_atoms
from pondera.data.tokenize import merge_and_encode, tokenize_dataset
from pondera.data.trace import ROLE_FINAL_MOVE, load_sidecar, synthesize_root
from pondera.models.checkpoint import load_model, read_checkpoint, save_checkpoint
from pondera.models.policy import create_model
from pondera.training.trace import TraceCollator, TrainConfig, trace_loss_parts, train

FIXTURES = Path(__file__).parent / "fixtures" / "trace_v2"
MODEL = {"num_blocks": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2}


class TextTests(unittest.TestCase):
    def test_text_requires_a_final_decision_and_value(self):
        prefix = "8/8/8/4k3/8/8/8/4K3 w - - 0 1 (repetition 3) "
        for suffix in ("FINAL END", "FINAL CLAIM_DRAW END"):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                trace_atoms(prefix + suffix)

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

    def test_configured_workflow_fits_one_vocabulary_size_and_encodes_all_rows(self):
        for vocab_size in (CHESS_VOCAB_SIZE, CHESS_VOCAB_SIZE + 8):
            with self.subTest(vocab_size=vocab_size):
                config = self.root / "data.toml"
                config.write_text(
                    Path("configs/data.toml")
                    .read_text()
                    .replace("vocab_size = 8192", f"vocab_size = {vocab_size}")
                    .replace("min_count = 8", "min_count = 1")
                    .replace("# max_roots = 128", "max_roots = 3")
                    .replace("batch_size = 128", "batch_size = 2")
                    .replace("train = 500000", "train = 0")
                    .replace("val = 2000", "val = 0")
                )
                output = self.root / f"training-{vocab_size}"
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
                self.dictionary = MergeDictionary(
                    load_manifest(output / "release.json")["merge_entries"]
                )
                self.assertLessEqual(
                    len(self.dictionary), vocab_size - CHESS_VOCAB_SIZE
                )
                if vocab_size > CHESS_VOCAB_SIZE:
                    self.assertGreater(len(self.dictionary), 0)
                self.assert_preserved(self.rows(output))
                document = json.loads((output / "profile" / "curve.json").read_text())
                self.assertEqual(document["settings"]["vocab_size"], vocab_size)
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
        for expected, profiled in zip(self.original, load_full_rows(self.source)):
            np.testing.assert_array_equal(profiled.atoms, expected["trace_token_ids"])
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

    def test_encoding_preserves_an_entirely_dropped_shard(self):
        directory = self.root / "drops"
        directory.mkdir()
        config, roots, run, sidecars = build_run(directory)
        dropped = {root["root_id"] for root in roots[:2]}

        def provider(root):
            if root["root_id"] in dropped:
                raise EngineTimeoutError("engine timed out")
            return sidecars[root["root_id"]]

        source = run / "release"
        produce_shards(
            roots, source, config, sidecar_for_root=provider, shard_size=2, workers=1
        )
        self.assertTrue(validate_release(source, roots, workers=1)["ok"])
        output = directory / "encoded"
        summary = tokenize_dataset(source, output, self.dictionary)
        self.assertEqual(summary["rows"], len(roots) - len(dropped))
        for path, rows, _ in parquet_sources(output):
            self.assertEqual(pq.ParquetFile(path).metadata.num_rows, rows)
        self.assertEqual(len(TracePositions(output, "train")), 3)
        again = directory / "encoded-again"
        tokenize_dataset(output, again, self.dictionary)
        self.assertEqual(self.rows(again), self.rows(output))

    def test_base_vocabulary_needs_no_full_presentation_fitting_rows(self):
        from dataclasses import replace

        for presentation in ("zero", "random"):
            (self.root / presentation).mkdir()
            config, roots, run, provider = build_run(
                self.root / presentation,
                cases=CASES,
                recipe_text=f"[recipe]\npresentations = {{{presentation} = 1.0}}",
            )
            source = run / "release"
            produce_shards(
                roots,
                source,
                config,
                sidecar_for_root=provider,
                shard_size=2,
                workers=1,
            )
            output = self.root / f"base-{presentation}"
            result = merge_and_encode(
                source,
                output,
                replace(load_config().merge, vocab_size=CHESS_VOCAB_SIZE),
            )
            self.assertEqual(result["entries"], 0)
            self.assertEqual(result["ratio"], 1.0)
            self.assertEqual(result["rows"], len(roots))
            profile = json.loads((output / "profile/curve.json").read_text())
            self.assertEqual(profile["corpus"]["fitting"]["rows"], 0)
            self.assertEqual(profile["candidates"]["counted"], 0)

    def test_hf_training_artifacts_have_distinct_persistent_identities(self):
        identities = []
        for index, row in enumerate(self.original[:2]):
            source = self.root / f"hf-{index}" / "data"
            source.mkdir(parents=True)
            pq.write_table(
                pa.Table.from_pylist([row], schema=HF_SCHEMA),
                source / "train-00000.parquet",
            )
            output = self.root / f"hf-encoded-{index}"
            tokenize_dataset(source.parent, output)
            identities.append(trace_data_identity(output))
            copied = self.root / f"reencoded-{index}"
            tokenize_dataset(output, copied)
            self.assertEqual(trace_data_identity(copied), identities[-1])
        self.assertNotEqual(identities[0], identities[1])

    def test_hf_import_rejects_a_different_root_header(self):
        original = self.original[0]
        for field, value in (("fen", self.original[1]["fen"]), ("repetition", 4)):
            with self.subTest(field=field):
                source = self.root / field / "data"
                source.mkdir(parents=True)
                pq.write_table(
                    pa.Table.from_pylist(
                        [{**original, field: value}], schema=HF_SCHEMA
                    ),
                    source / "train-00000.parquet",
                )
                with self.assertRaises(ValueError):
                    tokenize_dataset(source.parent, self.root / f"invalid-{field}")

    def test_library_conversion_rejects_unknown_split_labels(self):
        path = self.source / "shard-000000/data.parquet"
        table = pq.read_table(path)
        table = table.set_column(
            table.schema.get_field_index("split"),
            "split",
            pa.array(["validation"] * len(table)),
        )
        pq.write_table(table, path)
        with self.assertRaises(ValueError):
            tokenize_dataset(self.source, self.root / "invalid-split")

    def test_format_nine_base_cache_converts_to_inline_training_metadata(self):
        # The accepted producer input uses list<int16> base IDs in format 9.
        for path, _, _ in parquet_sources(self.source):
            table = pq.read_table(path)
            column = table.schema.get_field_index("trace_token_ids")
            table = table.set_column(
                column,
                "trace_token_ids",
                table["trace_token_ids"].cast(pa.list_(pa.int16())),
            )
            pq.write_table(table, path)
        manifest = load_manifest(self.source / "release.json")
        manifest["format_version"] = 9
        write_manifest(self.source / "release.json", manifest)
        original = load_full_rows(self.source)
        output = self.root / "converted"
        tokenize_dataset(self.source, output, self.dictionary, batch_size=2)
        self.assertEqual(
            load_manifest(output / "release.json")["format_version"],
            DATA_FORMAT_VERSION,
        )
        self.assert_preserved(self.rows(output))
        self.assertEqual(TracePositions(output, "train").dictionary, self.dictionary)
        converted = load_full_rows(output)
        for before, after in zip(original, converted):
            np.testing.assert_array_equal(before.atoms, after.atoms)

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
        self.assertEqual(data.dictionary, self.dictionary)
        self.assertEqual(
            load_manifest(output / "release.json")["merge_entries"],
            [list(entry) for entry in self.dictionary.entries],
        )
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
        manifest = load_manifest(output / "release.json")
        manifest["merge_entries"] = tuple(reversed(self.dictionary.entries))
        write_manifest(output / "release.json", manifest)
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
        manifest = load_manifest(data / "release.json")
        manifest["merge_entries"] = tuple(reversed(self.dictionary.entries))
        write_manifest(data / "release.json", manifest)
        with self.assertRaises(ValueError):
            train(config, data, output, resume=final, max_steps=2)

    def test_profile_preserves_splits_and_optional_empty_holdout(self):
        document, dictionaries = build_profile(
            [self.source],
            [CHESS_VOCAB_SIZE, CHESS_VOCAB_SIZE + 8],
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
            MergeDictionary(
                json.loads((output / "curve.json").read_text())["tiers"][1][
                    "merge_entries"
                ]
            ),
            dictionaries[CHESS_VOCAB_SIZE + 8],
        )
        self.assertEqual(
            json.loads((output / "curve.json").read_text())["corpus"]["loaded"]["rows"],
            5,
        )
        heldout, _ = build_profile(
            [self.source],
            [CHESS_VOCAB_SIZE + 8],
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
            [CHESS_VOCAB_SIZE + 8],
            min_count=1,
            per_length_limit=32,
            selection_stride=1,
            block=4,
            max_rounds=64,
        )
        self.assertEqual(
            repeats[CHESS_VOCAB_SIZE + 8], dictionaries[CHESS_VOCAB_SIZE + 8]
        )
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
        original = load_manifest(path)
        for field, value in (("grammar_version", 1), ("format_version", 10)):
            with self.subTest(field=field):
                write_manifest(path, {**original, field: value})
                with self.assertRaises(ValueError):
                    load_full_rows(self.source)
                with self.assertRaises(ValueError):
                    tokenize_dataset(
                        self.source, self.root / "invalid", self.dictionary
                    )
                with self.assertRaises(ValueError):
                    TracePositions(self.source, "train")


if __name__ == "__main__":
    unittest.main()
