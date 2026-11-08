"""Packed training streams: layout, batch contract, and sampler invariants."""

import hashlib
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from test_release import CASES, build_run

from pondera.chess.codec import decode_board, encode_board
from pondera.chess.vocabulary import CHESS_VOCAB_SIZE, FINAL, MOVE
from pondera.data.dataset import (
    load_manifest,
    parquet_sources,
    write_manifest,
    write_trace_rows,
)
from pondera.data.merge_profile import build_profile
from pondera.data.packing import (
    IGNORE_INDEX,
    MAX_DOCUMENT_TOKENS,
    PackedDocuments,
    assemble_batch,
    load_subsets,
    nested_subsets,
    pack_batches,
    packed_batch,
    write_packed,
)
from pondera.data.pipeline import produce_shards
from pondera.data.tokenize import tokenize_dataset


class PackPlanTests(unittest.TestCase):
    def test_greedy_whole_document_packing(self):
        # Hand-derived: lengths 3,1,4,1,5 under budget 6 close before the
        # document that would overflow, and the final batch is partial.
        batches, skipped = pack_batches([3, 1, 4, 1, 5], [0, 1, 2, 3, 4], 6)
        self.assertEqual(batches, [[0, 1], [2, 3], [4]])
        self.assertEqual(skipped, 0)

    def test_documents_longer_than_the_budget_are_skipped(self):
        batches, skipped = pack_batches([2, 10, 3], [0, 1, 2], 6)
        self.assertEqual(batches, [[0, 2]])
        self.assertEqual(skipped, 1)

    def test_the_drop_guardrail_applies_above_the_budget(self):
        batches, skipped = pack_batches(
            [2, MAX_DOCUMENT_TOKENS + 1, 3], [0, 1, 2], 10**9
        )
        self.assertEqual(batches, [[0, 2]])
        self.assertEqual(skipped, 1)

    def test_budget_must_be_positive(self):
        with self.assertRaises(ValueError):
            pack_batches([1], [0], 0)


class SubsetTests(unittest.TestCase):
    def test_nested_prefixes_and_contract_shuffle(self):
        subsets = nested_subsets(10, {"a": 2, "b": 5, "c": None}, 640)
        expected = list(range(10))
        random.Random("640:packing:subsets").shuffle(expected)
        self.assertEqual(subsets["c"], expected)
        self.assertEqual(subsets["b"], expected[:5])
        self.assertEqual(subsets["a"], expected[:2])
        self.assertEqual(subsets, nested_subsets(10, {"a": 2, "b": 5, "c": None}, 640))


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
        _, dictionaries = build_profile(
            [self.source],
            [CHESS_VOCAB_SIZE + 64],
            min_count=1,
            per_length_limit=64,
            block=4,
            max_rounds=64,
        )
        self.dictionary = dictionaries[CHESS_VOCAB_SIZE + 64]
        self.tokenized = self.root / "tokenized"
        tokenize_dataset(self.source, self.tokenized, self.dictionary)
        self.rows = [
            row
            for path, _, _ in parquet_sources(self.tokenized)
            for row in pq.read_table(path).to_pylist()
        ]
        self.expected = [
            np.concatenate(
                [
                    encode_board(row["fen"], row["repetition"]).astype(np.uint16),
                    np.asarray(row["trace_token_ids"], dtype=np.uint16),
                ]
            )
            for row in self.rows
        ]

    def pack(self):
        return write_packed(self.tokenized, self.root / "packed", seed=640)

    def test_documents_are_board_segment_then_trace_segment(self):
        self.pack()
        packed = PackedDocuments(self.root / "packed")
        self.assertEqual(len(packed), len(self.rows))
        for index, (row, expected) in enumerate(zip(self.rows, self.expected)):
            np.testing.assert_array_equal(packed.document(index), expected)
            board = encode_board(row["fen"], row["repetition"])
            np.testing.assert_array_equal(packed.document(index)[: len(board)], board)
            fen, repetition = decode_board(packed.document(index)[: len(board)])
            np.testing.assert_array_equal(encode_board(fen, repetition), board)

    def test_manifest_identity_splits_and_subsets(self):
        self.pack()
        packed = PackedDocuments(self.root / "packed")
        manifest = packed.manifest
        source = load_manifest(self.tokenized / "release.json")
        self.assertEqual(
            manifest["vocabulary"], CHESS_VOCAB_SIZE + len(self.dictionary)
        )
        self.assertEqual(manifest["source_dataset_id"], source["dataset_id"])
        self.assertEqual(manifest["documents"], len(self.rows))
        self.assertEqual(manifest["tokens"], sum(len(row) for row in self.expected))
        self.assertEqual(
            manifest["tokens_sha256"],
            hashlib.sha256(
                (self.root / "packed" / "tokens.u16").read_bytes()
            ).hexdigest(),
        )
        expected_splits = np.asarray(
            [0 if row["split"] == "train" else 1 for row in self.rows], np.uint8
        )
        np.testing.assert_array_equal(packed.split, expected_splits)
        train = [
            index for index, row in enumerate(self.rows) if row["split"] == "train"
        ]
        subsets = load_subsets(self.root / "packed")
        self.assertEqual(set(subsets), {"50k", "100k", "200k", "500k"})
        for indices in subsets.values():
            self.assertEqual(sorted(indices.tolist()), train)
        stats = json.loads((self.root / "packed" / "stats.json").read_text())
        self.assertEqual(stats["documents"], len(self.rows))
        self.assertEqual(stats["dropped_overlong"], 0)
        self.assertEqual(stats["splits"], {"train": len(train), "val": 1})

    def test_batches_follow_the_contract(self):
        self.pack()
        packed = PackedDocuments(self.root / "packed")
        indices = load_subsets(self.root / "packed")["500k"]
        budget = 4096
        # Contract shuffle (stdlib) restricted to packable documents.
        order = list(range(len(indices)))
        random.Random("640:packing:epoch:0").shuffle(order)
        packable = [
            int(indices[position])
            for position in order
            if packed.lengths[indices[position]] <= min(budget, MAX_DOCUMENT_TOKENS)
        ]
        content_to_index = {
            document.astype(np.int32).tobytes(): i
            for i, document in enumerate(self.expected)
        }
        self.assertEqual(len(content_to_index), len(self.expected))
        batches = []
        step = 0
        while True:
            batch = packed_batch(packed, indices, 640, budget, step)
            if batch["epoch"] != 0:
                break
            batches.append(batch)
            step += 1
        self.assertTrue(batches)
        # The epoch's batches partition the packable order into consecutive,
        # maximal, budget-bounded spans; membership is recovered by content.
        sequence = []
        for position, batch in enumerate(batches):
            ids, cu, labels = (
                batch["input_ids"],
                batch["cu_seqlens"],
                batch["labels"],
            )
            self.assertEqual(cu[0], 0)
            self.assertEqual(cu[-1], len(ids))
            self.assertLessEqual(len(ids), budget)
            members = []
            for start, stop in zip(cu, cu[1:]):
                members.append(content_to_index[ids[start:stop].tobytes()])
                np.testing.assert_array_equal(
                    labels[start : stop - 1], ids[start + 1 : stop]
                )
                self.assertEqual(labels[stop - 1], IGNORE_INDEX)
            self.assertEqual(int((labels == IGNORE_INDEX).sum()), batch["documents"])
            sequence.extend(members)
            if position + 1 < len(batches):
                following = packed.lengths[packable[len(sequence)]]
                self.assertGreater(len(ids) + following, budget)
        self.assertEqual(sequence, packable)

    def test_resume_and_determinism(self):
        self.pack()
        packed = PackedDocuments(self.root / "packed")
        indices = load_subsets(self.root / "packed")["500k"]
        first = packed_batch(packed, indices, 640, 2048, 0)
        again = packed_batch(packed, indices, 640, 2048, 0)
        self.assertEqual(first["input_ids"].tolist(), again["input_ids"].tolist())
        for step in (1, 2, 3):
            batch = packed_batch(packed, indices, 640, 2048, step)
            replayed = packed_batch(packed, indices, 640, 2048, step)
            self.assertEqual(batch["epoch"], replayed["epoch"])
            self.assertEqual(
                batch["input_ids"].tolist(), replayed["input_ids"].tolist()
            )

    def test_overlong_documents_are_counted_and_skipped(self):
        directory = self.root / "synthetic"
        rows = [
            {
                "fen": "8/8/8/4k3/8/8/8/4K3 w - - 0 1",
                "repetition": 1,
                "presentation": "full",
                "split": "train",
                "trace_token_ids": [MOVE + 1] * (MAX_DOCUMENT_TOKENS + 1),
                "trace_len": MAX_DOCUMENT_TOKENS + 1,
            },
            {
                "fen": "8/8/8/4k3/8/8/8/4K3 b - - 3 7",
                "repetition": 2,
                "presentation": "zero",
                "split": "train",
                "trace_token_ids": [MOVE + 2, MOVE + 3],
                "trace_len": 2,
            },
        ]
        write_trace_rows(directory / "shard-000000", rows)
        write_manifest(
            directory / "release.json",
            {
                "format_version": 11,
                "grammar_version": 2,
                "tokenizer_version": 3,
                "merge_entries": [],
                "shards": [{"path": "shard-000000", "rows": 2, "split": "train"}],
            },
        )
        output = self.root / "packed-synthetic"
        result = write_packed(directory, output, seed=640)
        self.assertEqual(result["dropped_overlong"], 1)
        packed = PackedDocuments(output)
        self.assertEqual(len(packed), 2)
        indices = load_subsets(output)["500k"]
        batches, skipped = pack_batches(packed.lengths, indices, 10**9)
        self.assertEqual(skipped, 1)
        self.assertEqual(len(batches), 1)
        kept = batches[0]
        self.assertEqual(
            int(packed.lengths[kept[0]]),
            len(encode_board(rows[1]["fen"], 2)) + 2,
        )

    def test_a_composite_containing_final_is_rejected(self):
        manifest = load_manifest(self.tokenized / "release.json")
        manifest["merge_entries"] = [[FINAL, MOVE + 1]]
        write_manifest(self.tokenized / "release.json", manifest)
        with self.assertRaises(ValueError):
            write_packed(self.tokenized, self.root / "packed-final", seed=640)

    def test_command_line_packs_a_tokenized_dataset(self):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pondera.data",
                "pack",
                "--input",
                str(self.tokenized),
                "--output",
                str(self.root / "packed-cli"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        packed = PackedDocuments(self.root / "packed-cli")
        self.assertEqual(len(packed), len(self.rows))

    def test_assemble_batch_shifts_labels_within_documents(self):
        self.pack()
        packed = PackedDocuments(self.root / "packed")
        batch = assemble_batch(packed, [0, 1])
        ids = batch["input_ids"]
        first, second = (packed.document(0), packed.document(1))
        np.testing.assert_array_equal(ids, np.concatenate([first, second]))
        self.assertEqual(batch["cu_seqlens"].tolist(), [0, len(first), len(ids)])
        labels = batch["labels"]
        self.assertEqual(labels[: len(first) - 1].tolist(), first[1:].tolist())
        self.assertEqual(labels[len(first) - 1], IGNORE_INDEX)
        self.assertEqual(labels[len(first) : -1].tolist(), second[1:].tolist())
        self.assertEqual(labels[-1], IGNORE_INDEX)


if __name__ == "__main__":
    unittest.main()
