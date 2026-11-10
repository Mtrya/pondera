"""Real teacher traces qualify authoritative text and merged training inputs."""

import json
import shutil
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

from pondera.chess.codec import encode_board
from pondera.chess.merge import MergeDictionary, MergeEncoder, decode
from pondera.chess.vocabulary import (
    BACK,
    BOARD,
    CHESS_VOCAB_SIZE,
    DELIM,
    DEPTH,
    END,
    FINAL,
    MOVE,
    REFUTED,
    RESET,
    VALUE,
)
from pondera.data.config import load_config
from pondera.data.dataset import (
    HF_SCHEMA,
    load_manifest,
    parquet_sources,
    write_manifest,
)
from pondera.data.merge_profile import (
    build_profile,
    load_full_rows,
    write_artifacts,
)
from pondera.data.packing import PackedDocuments
from pondera.data.pipeline import EngineTimeoutError, produce_shards, validate_release
from pondera.data.text import encode_trace, encode_traces, merge_segments, trace_atoms
from pondera.data.tokenize import encode_dataset, merge_and_encode
from pondera.data.trace import load_sidecar, synthesize_root

FIXTURES = Path(__file__).parent / "fixtures" / "trace_v2"


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
            atoms = row["trace_atom_ids"]
            final = atoms.index(FINAL)
            entries.append(tuple(atoms[final + 1 : final + 3]))
            entries.extend(
                tuple(segment[:8])
                for segment in merge_segments(atoms)
                if len(segment) > 1
            )
        self.dictionary = MergeDictionary(tuple(dict.fromkeys(entries)))

    @staticmethod
    def rows(directory):
        return [
            row
            for path, _, _ in parquet_sources(directory)
            for row in pq.read_table(path).to_pylist()
        ]

    def assert_encoded(self, directory, rows=None):
        rows = self.original if rows is None else rows
        packed = PackedDocuments(directory)
        self.assertEqual(len(packed), len(rows))
        for index, expected in enumerate(rows):
            np.testing.assert_array_equal(
                decode(packed.document(index), self.dictionary),
                np.concatenate(
                    [
                        encode_board(expected["fen"], expected["repetition"]),
                        expected["trace_atom_ids"],
                    ]
                ),
            )
        np.testing.assert_array_equal(
            packed.split, [0 if row["split"] == "train" else 1 for row in rows]
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
                    load_manifest(output / "packing.json")["merge_entries"]
                )
                self.assertLessEqual(
                    len(self.dictionary), vocab_size - CHESS_VOCAB_SIZE
                )
                if vocab_size > CHESS_VOCAB_SIZE:
                    self.assertGreater(len(self.dictionary), 0)
                self.assert_encoded(output)
                document = json.loads((output / "profile" / "curve.json").read_text())
                self.assertEqual(document["settings"]["vocab_size"], vocab_size)
                self.assertEqual(document["corpus"]["loaded"]["rows"], 3)
                self.assertEqual(len(document["tiers"]), 1)
                self.assertGreater(document["timing"]["workflow_peak_rss_bytes"], 0)
                self.assertEqual(int((PackedDocuments(output).split == 0).sum()), 4)

    def test_encoding_uses_authoritative_text_despite_corrupt_atom_cache(self):
        path = self.source / "shard-000000" / "data.parquet"
        table = pq.read_table(path)
        corrupt = [[CHESS_VOCAB_SIZE + 100]] * len(table)
        table = table.set_column(
            table.schema.get_field_index("trace_atom_ids"),
            "trace_atom_ids",
            pa.array(corrupt, type=pa.list_(pa.int32())),
        )
        pq.write_table(table, path)
        for expected, profiled in zip(self.original, load_full_rows(self.source)):
            np.testing.assert_array_equal(profiled.atoms, expected["trace_atom_ids"])
        output = self.root / "encoded"
        result = encode_dataset(self.source, output, self.dictionary, batch_size=1)
        self.assertEqual(result["documents"], len(self.original))
        self.assertLess(result["trace_tokens"], result["base_atoms"])
        self.assert_encoded(output)

    def test_encoding_handles_an_entirely_dropped_shard(self):
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
        summary = encode_dataset(source, output, self.dictionary)
        self.assertEqual(summary["documents"], len(roots) - len(dropped))
        self.assertEqual(int((PackedDocuments(output).split == 0).sum()), 3)

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
            self.assertEqual(result["documents"], len(roots))
            profile = json.loads((output / "profile/curve.json").read_text())
            self.assertEqual(profile["corpus"]["fitting"]["rows"], 0)
            self.assertEqual(profile["candidates"]["counted"], 0)

    def test_fit_checkpoint_drives_encoding_and_guards_settings(self):
        from dataclasses import replace

        config = replace(load_config().merge, vocab_size=CHESS_VOCAB_SIZE)
        output = self.root / "checkpointed"
        first = merge_and_encode(self.source, output, config)
        self.assertEqual(first["entries"], 0)

        checkpoint = output.parent / f"{output.name}.fit-checkpoint.json"
        record = json.loads(checkpoint.read_text())
        self.assertEqual(record["settings"]["vocab_size"], CHESS_VOCAB_SIZE)
        self.assertEqual(record["entries"], [])
        self.assertEqual(load_manifest(output / "packing.json")["merge_entries"], [])

        # A rerun trusts the checkpoint's dictionary instead of refitting: a
        # tampered-but-valid entry set shows up in the encoded release.
        record["entries"] = [list(self.dictionary.entries[0])]
        checkpoint.write_text(json.dumps(record) + "\n")
        shutil.rmtree(output)
        resumed = merge_and_encode(self.source, output, config)
        self.assertEqual(resumed["entries"], 1)
        self.assertEqual(
            load_manifest(output / "packing.json")["merge_entries"],
            [list(self.dictionary.entries[0])],
        )

        # Settings that would change the fit reject the checkpoint.
        record["settings"]["min_count"] += 1
        checkpoint.write_text(json.dumps(record) + "\n")
        shutil.rmtree(output)
        with self.assertRaises(ValueError):
            merge_and_encode(self.source, output, config)
        self.assertFalse(output.exists())

    def test_fit_checkpoint_rejects_a_different_source_or_release(self):
        from dataclasses import replace

        config = replace(load_config().merge, vocab_size=CHESS_VOCAB_SIZE)
        output = self.root / "source-checkpoint"
        merge_and_encode(self.source, output, config)
        shutil.rmtree(output)
        other = self.root / "other-source"
        shutil.copytree(self.source, other)
        with self.assertRaises(ValueError):
            merge_and_encode(other, output, config)
        manifest = load_manifest(self.source / "release.json")
        manifest["dataset_id"] = "replacement-dataset"
        write_manifest(self.source / "release.json", manifest)
        with self.assertRaises(ValueError):
            merge_and_encode(self.source, output, config)
        self.assertFalse(output.exists())

    def test_encode_rejects_vocabulary_budgets_outside_packed_range(self):
        for size in (CHESS_VOCAB_SIZE - 1, 65536):
            with self.subTest(size=size):
                config = self.root / "invalid-vocabulary.toml"
                config.write_text(
                    Path("configs/data.toml")
                    .read_text()
                    .replace("vocab_size = 8192", f"vocab_size = {size}")
                )
                output = self.root / f"invalid-{size}"
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pondera.data",
                        "encode",
                        "--input",
                        str(self.source),
                        "--output",
                        str(output),
                        "--config",
                        str(config),
                    ],
                    text=True,
                    capture_output=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())

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
            encode_dataset(source.parent, output)
            identities.append(
                load_manifest(output / "packing.json")["source_dataset_id"]
            )
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
                    encode_dataset(source.parent, self.root / f"invalid-{field}")

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
            encode_dataset(self.source, self.root / "invalid-split")

    def test_existing_curated_text_is_reusable_without_cache_migration(self):
        # Old local field names do not affect encoding; only text is read.
        for path, _, _ in parquet_sources(self.source):
            table = pq.read_table(path)
            table = table.rename_columns(
                [
                    {
                        "trace_atom_ids": "trace_token_ids",
                        "trace_atom_count": "trace_len",
                    }.get(name, name)
                    for name in table.column_names
                ]
            )
            pq.write_table(table, path)
        manifest = load_manifest(self.source / "release.json")
        manifest["format_version"] = 11
        write_manifest(self.source / "release.json", manifest)
        output = self.root / "reused"
        encode_dataset(self.source, output, self.dictionary)
        self.assert_encoded(output)

    def test_hf_input_encodes_text_and_preserves_split_membership(self):
        exported = self.root / "hf"
        export_hf(self.source, exported, batch_size=2)
        output = self.root / "hf-encoded"
        encode_dataset(exported, output, self.dictionary, batch_size=1)
        expected = [row for row in self.original if row["split"] == "train"]
        expected += [row for row in self.original if row["split"] == "val"]
        self.assert_encoded(output, expected)
        self.assertEqual(int((PackedDocuments(output).split == 0).sum()), 4)
        self.assertEqual(int((PackedDocuments(output).split == 1).sum()), 1)
        # The shared HF schema stays text-only.
        for path, _, _ in parquet_sources(exported):
            self.assertEqual(pq.ParquetFile(path).schema_arrow, HF_SCHEMA)

    def test_fitting_keeps_all_controls_standalone(self):
        document, dictionaries = build_profile(
            [self.source],
            [CHESS_VOCAB_SIZE + 64],
            min_count=1,
            per_length_limit=64,
            block=4,
            max_rounds=64,
        )
        dictionary = dictionaries[CHESS_VOCAB_SIZE + 64]
        self.assertGreater(len(dictionary), 0)
        for entry in dictionary.entries:
            self.assertTrue(all(atom < DELIM for atom in entry))
        encoder = MergeEncoder(dictionary)
        for row in self.original:
            atoms = row["trace_atom_ids"]
            self.assertEqual(atoms.count(FINAL), 1)
            encoded = encode_trace(atoms, encoder)
            self.assertEqual(int(np.count_nonzero(encoded == FINAL)), 1)

    def test_final_decision_is_standalone_even_with_a_matching_pair(self):
        atoms = np.array(
            [DEPTH, MOVE + 1, VALUE + 1, BACK, FINAL, MOVE + 1, VALUE + 1, END],
            np.int32,
        )
        dictionary = MergeDictionary([(MOVE + 1, VALUE + 1)])
        expected = [DEPTH, CHESS_VOCAB_SIZE, BACK, FINAL, MOVE + 1, VALUE + 1, END]
        np.testing.assert_array_equal(
            encode_trace(atoms, MergeEncoder(dictionary)), expected
        )
        np.testing.assert_array_equal(encode_traces([atoms], dictionary)[0], expected)
        np.testing.assert_array_equal(decode(expected, dictionary), atoms)

    def test_each_control_is_rejected_in_an_external_dictionary(self):
        for atom in (BACK, RESET, BOARD, FINAL, REFUTED, DEPTH, END):
            with self.subTest(atom=atom), self.assertRaises(ValueError):
                encode_dataset(
                    self.source,
                    self.root / "bad-dictionary",
                    MergeDictionary([(atom, MOVE + 1)]),
                )

    def test_parallel_profile_and_tokenization_match_sequential(self):
        document, dictionaries = build_profile(
            [self.source],
            [CHESS_VOCAB_SIZE + 8],
            min_count=1,
            per_length_limit=32,
            block=4,
            max_rounds=64,
        )
        parallel_document, parallel_dictionaries = build_profile(
            [self.source],
            [CHESS_VOCAB_SIZE + 8],
            min_count=1,
            per_length_limit=32,
            block=4,
            max_rounds=64,
            workers=2,
        )
        self.assertEqual(parallel_dictionaries, dictionaries)

        def timeless(profile):
            stripped = {key: value for key, value in profile.items() if key != "timing"}
            stripped["tiers"] = [
                {
                    key: value
                    for key, value in tier.items()
                    if not key.endswith("_seconds")
                }
                for tier in stripped["tiers"]
            ]
            return stripped

        self.assertEqual(timeless(parallel_document), timeless(document))
        serial = self.root / "serial"
        encode_dataset(self.source, serial, self.dictionary, batch_size=1)
        parallel = self.root / "parallel"
        encode_dataset(self.source, parallel, self.dictionary, batch_size=1, workers=2)
        self.assert_encoded(serial)
        self.assert_encoded(parallel)
        for name in ("tokens.u16", "offsets.i64", "split.u8", "presentation.u8"):
            self.assertEqual(
                (parallel / name).read_bytes(), (serial / name).read_bytes()
            )
        self.assertEqual(
            load_manifest(parallel / "packing.json"),
            load_manifest(serial / "packing.json"),
        )

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

    def test_input_rejects_mismatched_trace_versions(self):
        path = self.source / "release.json"
        original = load_manifest(path)
        for field, value in (("grammar_version", 1), ("tokenizer_version", 1)):
            with self.subTest(field=field):
                write_manifest(path, {**original, field: value})
                with self.assertRaises(ValueError):
                    load_full_rows(self.source)
                with self.assertRaises(ValueError):
                    encode_dataset(self.source, self.root / "invalid", self.dictionary)


if __name__ == "__main__":
    unittest.main()
