"""Memory-map layout, batch contract, and consumed-document sampler invariants."""

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from test_release import CASES, build_run

from pondera.chess.codec import decode_board, encode_board
from pondera.chess.merge import MergeDictionary
from pondera.data.dataset import (
    TRACE_SCHEMA,
    load_manifest,
    parquet_sources,
    write_manifest,
)
from pondera.data.packing import (
    IGNORE_INDEX,
    PackedDocuments,
    assemble_batch,
    pack_batches,
    packed_batch_indices,
    packed_batches,
    select_documents,
)
from pondera.data.pipeline import produce_shards
from pondera.data.tokenize import encode_dataset


class PackPlanTests(unittest.TestCase):
    def test_greedy_whole_document_packing(self):
        # Hand-derived: close before the next document would overflow budget 6.
        self.assertEqual(
            pack_batches([3, 1, 4, 1, 5], range(5), 6), [[0, 1], [2, 3], [4]]
        )

    def test_documents_exceeding_the_budget_raise(self):
        with self.assertRaises(ValueError):
            pack_batches([2, 10, 3], range(3), 6)

    def test_budget_must_be_positive(self):
        with self.assertRaises(ValueError):
            pack_batches([1], [0], 0)

    def test_extending_a_partial_batch_target_consumes_every_document(self):
        # Four two-token documents fit three per batch. A half-presentation
        # stops after two; resuming must return both remaining documents.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.array([10, 11, 20, 21, 30, 31, 40, 41], np.uint16).tofile(
                root / "tokens.u16"
            )
            np.array([0, 2, 4, 6, 8], np.int64).tofile(root / "offsets.i64")
            np.zeros(4, np.uint8).tofile(root / "split.u8")
            np.zeros(4, np.uint8).tofile(root / "presentation.u8")
            write_manifest(
                root / "packing.json", {"format_version": 3, "max_document_tokens": 10}
            )
            documents = PackedDocuments(root)
            indices = np.arange(4)
            partial = list(
                packed_batch_indices(documents, indices, 640, 6, presentations=0.5)
            )
            resumed = list(
                packed_batch_indices(
                    documents, indices, 640, 6, start_document=2, presentations=1
                )
            )
            full = list(
                packed_batch_indices(documents, indices, 640, 6, presentations=1)
            )
            actual = [int(i) for batch, _ in partial + resumed for i in batch]
            expected = [int(i) for batch, _ in full for i in batch]
            self.assertEqual(actual, expected)
            self.assertEqual(sorted(actual), [0, 1, 2, 3])
            self.assertEqual([len(batch) for batch, _ in full], [3, 1])
            self.assertEqual(
                list(
                    packed_batch_indices(
                        documents, indices, 640, 6, start_document=4, presentations=1
                    )
                ),
                [],
            )


class PackingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        config, roots, run, provider = build_run(
            self.root, cases=CASES, recipe_text="[recipe]\npresentations = {full = 1.0}"
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
        self.rows = [
            row
            for path, _, _ in parquet_sources(self.source)
            for row in pq.read_table(path).to_pylist()
        ]
        self.expected = [
            np.concatenate(
                [encode_board(row["fen"], row["repetition"]), row["trace_atom_ids"]]
            )
            for row in self.rows
        ]

    def pack(self, **kwargs):
        return encode_dataset(self.source, self.root / "packed", **kwargs)

    def test_documents_are_board_then_trace_and_preserve_identity(self):
        self.pack()
        packed = PackedDocuments(self.root / "packed")
        self.assertEqual(len(packed), len(self.rows))
        for index, (row, expected) in enumerate(zip(self.rows, self.expected)):
            np.testing.assert_array_equal(packed.document(index), expected)
            board = encode_board(row["fen"], row["repetition"])
            fen, repetition = decode_board(packed.document(index)[: len(board)])
            np.testing.assert_array_equal(encode_board(fen, repetition), board)
        self.assertEqual(
            packed.manifest["source_dataset_id"],
            load_manifest(self.source / "release.json")["dataset_id"],
        )
        self.assertEqual(
            packed.manifest["vocabulary"], len(MergeDictionary().token_lengths)
        )
        self.assertEqual(packed.manifest["tokens"], sum(map(len, self.expected)))
        np.testing.assert_array_equal(
            packed.split, [0 if row["split"] == "train" else 1 for row in self.rows]
        )
        stats = json.loads((self.root / "packed" / "stats.json").read_text())
        self.assertEqual(stats["documents"], len(self.rows))
        self.assertEqual(stats["ineligible_documents"], 0)

    def test_selection_is_nested_and_cap_can_change_without_reencoding(self):
        cap = sorted(
            len(doc)
            for doc, row in zip(self.expected, self.rows)
            if row["split"] == "train"
        )[0]
        self.pack(max_document_tokens=cap)
        packed = PackedDocuments(self.root / "packed")
        eligible = select_documents(packed, seed=640)
        all_train = select_documents(
            packed, seed=640, max_document_tokens=int(packed.lengths.max())
        )
        smaller = select_documents(
            packed, seed=640, max_document_tokens=int(packed.lengths.max()), count=2
        )
        np.testing.assert_array_equal(smaller, all_train[:2])
        self.assertEqual(
            set(eligible),
            {
                i
                for i, row in enumerate(self.rows)
                if row["split"] == "train" and len(self.expected[i]) <= cap
            },
        )
        self.assertLess(len(eligible), len(all_train))
        self.assertEqual(len(packed), len(self.rows))
        for index, expected in enumerate(self.expected):
            np.testing.assert_array_equal(packed.document(index), expected)
        stats = load_manifest(self.root / "packed" / "stats.json")
        self.assertEqual(
            stats["ineligible_documents"], sum(len(doc) > cap for doc in self.expected)
        )

    def test_documents_survive_arrow_batch_boundaries(self):
        source = self.root / "many-rows"
        source.mkdir()
        rows = self.rows * 20
        pq.write_table(
            pa.Table.from_pylist(rows, schema=TRACE_SCHEMA),
            source / "data.parquet",
            row_group_size=17,
        )
        manifest = load_manifest(self.source / "release.json")
        manifest["shards"] = [{"path": "data.parquet", "rows": len(rows)}]
        write_manifest(source / "release.json", manifest)
        output = self.root / "many-packed"
        encode_dataset(source, output, batch_size=64)
        packed = PackedDocuments(output)
        self.assertEqual(len(packed), len(rows))
        for index in range(len(rows)):
            np.testing.assert_array_equal(
                packed.document(index), self.expected[index % len(self.expected)]
            )

    def test_incomplete_shard_is_not_published(self):
        manifest = load_manifest(self.source / "release.json")
        manifest["shards"][0]["rows"] += 1
        write_manifest(self.source / "release.json", manifest)
        output = self.root / "incomplete"
        with self.assertRaises(ValueError):
            encode_dataset(self.source, output)
        self.assertFalse(output.exists())

    def test_batches_follow_the_contract(self):
        self.pack()
        packed = PackedDocuments(self.root / "packed")
        indices = select_documents(packed, seed=640)
        budget = int(packed.lengths.max()) + 128
        order = list(range(len(indices)))
        random.Random("640:packing:epoch:0").shuffle(order)
        expected_order = indices[order].tolist()
        content_to_index = {
            document.astype(np.int32).tobytes(): i
            for i, document in enumerate(self.expected)
        }
        sequence = []
        batches = list(packed_batches(packed, indices, 640, budget, presentations=1))
        for position, batch in enumerate(batches):
            ids, cu, labels = batch["input_ids"], batch["cu_seqlens"], batch["labels"]
            self.assertEqual(ids.dtype, np.int32)
            self.assertEqual(cu.dtype, np.int32)
            self.assertEqual(labels.dtype, np.int32)
            self.assertEqual(cu[0], 0)
            self.assertEqual(cu[-1], len(ids))
            self.assertLessEqual(len(ids), budget)
            for start, stop in zip(cu, cu[1:]):
                sequence.append(content_to_index[ids[start:stop].tobytes()])
                np.testing.assert_array_equal(
                    labels[start : stop - 1], ids[start + 1 : stop]
                )
                self.assertEqual(labels[stop - 1], IGNORE_INDEX)
            self.assertEqual(int((labels == IGNORE_INDEX).sum()), batch["documents"])
            if position + 1 < len(batches):
                self.assertGreater(
                    len(ids) + packed.lengths[expected_order[len(sequence)]], budget
                )
        self.assertEqual(sequence, expected_order)

    def test_resume_at_completed_batch_matches_uninterrupted_sequence(self):
        self.pack()
        packed = PackedDocuments(self.root / "packed")
        indices = select_documents(packed, seed=640)
        budget = int(packed.lengths.max()) + 128
        full = list(packed_batches(packed, indices, 640, budget, presentations=2.5))
        consumed = sum(batch["documents"] for batch in full[:2])
        resumed = list(
            packed_batches(
                packed, indices, 640, budget, start_document=consumed, presentations=2.5
            )
        )
        self.assertEqual(len(resumed), len(full) - 2)
        for actual, expected in zip(resumed, full[2:]):
            np.testing.assert_array_equal(actual["input_ids"], expected["input_ids"])
            self.assertEqual(actual["epoch"], expected["epoch"])

    def test_selected_document_budget_is_checked_without_skipping(self):
        self.pack()
        packed = PackedDocuments(self.root / "packed")
        indices = select_documents(packed, seed=640)
        maximum = int(packed.lengths[indices].max())
        with self.assertRaises(ValueError):
            next(packed_batches(packed, indices, 640, maximum - 1))
        self.assertEqual(
            sum(
                batch["documents"]
                for batch in packed_batches(
                    packed, indices, 640, maximum, presentations=1
                )
            ),
            len(indices),
        )

    def test_assemble_batch_shifts_labels_within_documents(self):
        self.pack()
        packed = PackedDocuments(self.root / "packed")
        batch = assemble_batch(packed, [0, 1])
        first, second = packed.document(0), packed.document(1)
        np.testing.assert_array_equal(
            batch["input_ids"], np.concatenate([first, second])
        )
        self.assertEqual(
            batch["cu_seqlens"].tolist(), [0, len(first), len(first) + len(second)]
        )
        np.testing.assert_array_equal(batch["labels"][: len(first) - 1], first[1:])
        np.testing.assert_array_equal(batch["labels"][len(first) : -1], second[1:])
        self.assertEqual(batch["labels"][len(first) - 1], IGNORE_INDEX)
        self.assertEqual(batch["labels"][-1], IGNORE_INDEX)


if __name__ == "__main__":
    unittest.main()
