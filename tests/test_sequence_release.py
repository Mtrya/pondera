import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import chess
import numpy as np

from pondera.data.positions import SequencePositions
from pondera.data.sequence_release import (
    SequenceReleaseConfig,
    merge_releases,
    produce_shards,
)
from pondera.data.sequence_validation import validate_release
from pondera.data.trace import replay
from pondera.training.sequence import SequenceTrainConfig
from pondera.training.sequence import train as train_sequence


@unittest.skipUnless(shutil.which("stockfish"), "Requires Stockfish on PATH")
class SequenceReleaseTests(unittest.TestCase):
    def test_sharded_release_roundtrip_and_resume(self):
        board = chess.Board()
        board.push_uci("e2e4")
        roots = [
            {
                "root_id": "root-a",
                "split": "train",
                "fen": chess.STARTING_FEN,
                "repetition": 1,
                "turn": "white",
                "source": {},
                "d16": {"bm": "e2e4", "value": 0.1, "mate": None},
            },
            {
                "root_id": "root-b",
                "split": "val",
                "fen": board.fen(en_passant="fen"),
                "repetition": 1,
                "turn": "black",
                "source": {},
                "d16": {"bm": "e7e5", "value": 0.1, "mate": None},
            },
        ]
        config = SequenceReleaseConfig(workers=2)
        metadata = {"teachers": ["fixed-depth-3-width-2", "shallow-stability"]}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = produce_shards(roots, output, config, metadata, shard_size=1)
            self.assertEqual(result["roots"], 2)
            self.assertEqual(len(result["shards"]), 2)
            train = SequencePositions(output, "train")
            val = SequencePositions(output, "val")
            self.assertEqual(len(train), 2)
            self.assertEqual(len(val), 2)
            self.assertEqual(
                set(train.sharded.arrays),
                {"board_tokens", "trace_tokens", "trace_len", "trace_roles"},
            )
            validation = validate_release(output, roots, workers=2)
            self.assertEqual(validation["rows"], 4)
            self.assertEqual(validation["roots"], 2)
            self.assertGreater(validation["states"], 0)
            for index in range(len(train)):
                board, trace, length, roles = train[index]
                program = replay(trace[:length], chess.STARTING_FEN)
                self.assertEqual(len(roles[:length]), length)
                self.assertEqual(program.final_move_index, length - 1)
                self.assertEqual(roles[length - 1], 5)
            for index in range(len(val)):
                _, trace, length, roles = val[index]
                board = chess.Board()
                board.push_uci("e2e4")
                replay(trace[:length], board.fen(en_passant="fen"))
                self.assertEqual(roles[length - 1], 5)
            resumed = produce_shards(roots, output, config, metadata, shard_size=1)
            self.assertEqual(resumed["roots"], 2)
            self.assertEqual(len(resumed["shards"]), 2)
            with self.assertRaises(ValueError):
                produce_shards(
                    roots, output, replace(config, depth=9), metadata, shard_size=1
                )
            changed_roots = [dict(root) for root in roots]
            changed_roots[0]["d16"] = {**roots[0]["d16"], "value": 0.2}
            with self.assertRaises(ValueError):
                produce_shards(changed_roots, output, config, metadata, shard_size=1)
            with self.assertRaises(ValueError):
                produce_shards(
                    roots,
                    output,
                    config,
                    {"teachers": ["fixed-depth-4-width-2"]},
                    shard_size=1,
                )

            recovered = output / "recovered"
            produce_shards(roots[:1], recovered, config, metadata, shard_size=1)
            first_trace = (recovered / "shard-000000" / "trace_tokens.npy").read_bytes()
            (recovered / "release.json").unlink()
            restored = produce_shards(roots, recovered, config, metadata, shard_size=1)
            self.assertEqual(restored["roots"], 2)
            self.assertEqual(len(restored["shards"]), 2)
            self.assertEqual(
                first_trace,
                (recovered / "shard-000000" / "trace_tokens.npy").read_bytes(),
            )

            left = output / "left"
            right = output / "right"
            for directory, root, start in ((left, roots[0], 0), (right, roots[1], 1)):
                part_metadata = {
                    **metadata,
                    "root_selection": {"start": start, "end": start + 1},
                }
                produce_shards([root], directory, config, part_metadata, shard_size=1)
            merged = output / "merged"
            merge_releases([left, right], merged)
            merged_manifest = json.loads((merged / "release.json").read_text())
            self.assertEqual(len(merged_manifest["complete_root_ids"]), 2)
            self.assertEqual(
                merged_manifest["metadata"]["root_selections"],
                [{"start": 0, "end": 1}, {"start": 1, "end": 2}],
            )
            self.assertEqual(len(merged_manifest["metadata"]["root_sets"]), 2)
            self.assertNotIn("roots", merged_manifest["metadata"])
            self.assertEqual(len(SequencePositions(merged, "train")), 2)
            self.assertEqual(len(SequencePositions(merged, "val")), 2)
            with self.assertRaises(ValueError):
                merge_releases([left, left], output / "duplicates")
            self.assertFalse((output / "duplicates").exists())

            training_config = SequenceTrainConfig(
                model={
                    "architecture": "sequence",
                    "num_blocks": 1,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_heads": 2,
                },
                batch_size=2,
                accumulation_steps=1,
                epochs=1,
                workers=0,
                precision="fp32",
                val_every=1,
                log_every=1,
                save_every=1,
            )
            train_sequence(training_config, output, output / "training", max_steps=1)
            metrics = [
                json.loads(line)
                for line in (output / "training" / "metrics.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(metrics[-1]["step"], 1)
            self.assertTrue((output / "training" / "final.pth").is_file())

            role_path = left / "shard-000000" / "trace_roles.npy"
            role_array = np.load(role_path, mmap_mode="r+")
            original_role = int(role_array[0, 0])
            role_array[0, 0] = 127
            role_array.flush()
            del role_array
            with self.assertRaises(ValueError):
                SequencePositions(left, "train")
            role_array = np.load(role_path, mmap_mode="r+")
            role_array[0, 0] = original_role
            role_array.flush()
            del role_array

            length_path = left / "shard-000000" / "trace_len.npy"
            length_array = np.load(length_path, mmap_mode="r+")
            original_length = int(length_array[0])
            length_array[0] = 0
            length_array.flush()
            del length_array
            with self.assertRaises(ValueError):
                SequencePositions(left, "train")
            length_array = np.load(length_path, mmap_mode="r+")
            length_array[0] = original_length
            length_array.flush()
            del length_array

            for directory, shard_path in (
                (left, left / "shard-000000" / "trace_tokens.npy"),
                (output, output / "shard-000000" / "trace_tokens.npy"),
            ):
                with shard_path.open("r+b") as handle:
                    handle.seek(-1, 2)
                    last = handle.read(1)
                    handle.seek(-1, 2)
                    handle.write(bytes([last[0] ^ 1]))
                if directory == left:
                    with self.assertRaises(ValueError):
                        merge_releases([left, right], output / "tampered-merge")
                    self.assertFalse((output / "tampered-merge").exists())
                else:
                    with self.assertRaises(ValueError):
                        produce_shards(roots, output, config, metadata, shard_size=1)


if __name__ == "__main__":
    unittest.main()
