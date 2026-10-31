import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

import chess
from teacher_release import depth_row, write_depth_release

from pondera.data.sequence_pilot import (
    JsonlStore,
    PilotConfig,
    build_summary,
    ensure_roots,
    select_roots,
)

TEACHER = "fixed-depth-3-width-2"


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def move_after(moves):
    board = chess.Board()
    for move in moves:
        board.push_uci(move)
    return board.fen(en_passant="fen"), sorted(
        move.uci() for move in board.legal_moves
    )[0]


def eligible_row(moves, value=0.1):
    fen, move = move_after(moves)
    return depth_row(fen, [(16, move, None, value)])


class RootSelectionTests(unittest.TestCase):
    def build_release(self, root):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        train = [
            eligible_row(["e2e4"]),
            eligible_row(["d2d4"]),
            eligible_row(["g1f3"]),
            eligible_row(["b1c3"]),
            {"fen": "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "terminal": True},
        ]
        # A duplicate of the first eligible position, then an incomplete row,
        # then a row whose key is selected in validation.
        duplicate = dict(train[0])
        train.append(duplicate)
        incomplete_fen, incomplete_move = move_after(["d2d4", "d7d5", "c2c4"])
        train.append(depth_row(incomplete_fen, [(15, incomplete_move, None, 0.0)]))
        overlap = eligible_row(["e2e4", "e7e5"])
        train.append(overlap)
        val = [
            overlap,
            eligible_row(["e2e4", "e7e5", "g1f3", "g8f6"]),
        ]
        manifest = write_depth_release(root, {"train": train, "val": val})
        return manifest, train, val

    def config(self, manifest, **overrides):
        values = {
            "release": str(manifest),
            "release_sha256": file_sha(manifest),
            "train_roots": 4,
            "val_roots": 2,
            "seed": 640,
            "teachers": (TEACHER,),
        }
        values.update(overrides)
        return PilotConfig(**values)

    def test_selection_filters_counts_and_keeps_splits_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest, train, val = self.build_release(directory / "release")
            config = self.config(manifest)
            roots, selection = select_roots(config, directory / "out")
            self.assertEqual(len(roots), 6)
            self.assertEqual(selection["counts"]["train"]["rows"], len(train))
            self.assertEqual(selection["counts"]["val"]["rows"], len(val))
            self.assertEqual(selection["counts"]["train"]["eligible"], 4)
            self.assertEqual(selection["counts"]["train"]["selected"], 4)
            self.assertEqual(selection["counts"]["val"]["eligible"], 2)
            self.assertEqual(selection["counts"]["val"]["selected"], 2)
            self.assertEqual(
                selection["counts"]["train"]["excluded"],
                {
                    "terminal": 1,
                    "incomplete_reference": 1,
                    "duplicate_position": 1,
                    "split_overlap": 1,
                },
            )
            self.assertEqual(selection["counts"]["val"]["excluded"], {})
            train_roots = [root for root in roots if root["split"] == "train"]
            val_roots = [root for root in roots if root["split"] == "val"]
            self.assertEqual(len(train_roots), 4)
            self.assertEqual(len(val_roots), 2)
            train_keys = {root["position_key"] for root in train_roots}
            val_keys = {root["position_key"] for root in val_roots}
            self.assertFalse(train_keys & val_keys)
            self.assertEqual(len({root["root_id"] for root in roots}), 6)
            global_rows = [
                (root["split"], root["source"]["global_row"]) for root in roots
            ]
            self.assertEqual(len(global_rows), len(set(global_rows)))
            for root in roots:
                self.assertGreaterEqual(root["source"]["global_row"], 0)
                self.assertGreaterEqual(root["source"]["row"], 0)
                self.assertEqual(
                    root["position_key"],
                    " ".join(root["fen"].split()[:4]),
                )
                board = chess.Board(root["fen"])
                self.assertFalse(board.is_game_over())
                self.assertIn(
                    chess.Move.from_uci(root["d16"]["bm"]),
                    board.legal_moves,
                )
            manifest_path = directory / "out" / "roots.manifest.json"
            payload = json.loads(manifest_path.read_text())
            self.assertEqual(payload["selection"]["seed"], 640)
            self.assertEqual(
                payload["files"]["roots.jsonl"]["sha256"],
                file_sha(directory / "out" / "roots.jsonl"),
            )
            self.assertEqual(payload["release"]["sha256"], config.release_sha256)

    def test_selection_is_deterministic_across_output_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest, _, _ = self.build_release(directory / "release")
            config = self.config(manifest, train_roots=2, val_roots=1)
            first, _ = select_roots(config, directory / "first")
            second, _ = select_roots(config, directory / "second")
            self.assertEqual(
                (directory / "first" / "roots.jsonl").read_bytes(),
                (directory / "second" / "roots.jsonl").read_bytes(),
            )
            self.assertEqual(first, second)
            self.assertEqual(len(first), 3)

    def test_release_and_shard_checksums_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest, _, _ = self.build_release(directory / "release")
            wrong = self.config(manifest, release_sha256="0" * 64)
            with self.assertRaises(ValueError):
                select_roots(wrong, directory / "wrong")

            shard = manifest.parent / "train.jsonl"
            original = shard.read_bytes()
            self.assertIn(b'"value": 0.1', original)
            shard.write_bytes(original.replace(b'"value": 0.1', b'"value": 0.2', 1))
            with self.assertRaises(ValueError):
                select_roots(self.config(manifest), directory / "tampered")

    def test_resume_reuses_and_rejects_tampered_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest, _, _ = self.build_release(directory / "release")
            config = self.config(manifest)
            first, _ = ensure_roots(config, directory / "out")
            second, _ = ensure_roots(config, directory / "out")
            self.assertEqual(first, second)
            roots_path = directory / "out" / "roots.jsonl"
            roots_path.write_bytes(roots_path.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                ensure_roots(config, directory / "out")
            different = self.config(manifest, seed=641)
            with self.assertRaises(ValueError):
                ensure_roots(different, directory / "out")

    def test_small_counts_select_only_the_requested_number_of_eligible_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest, _, _ = self.build_release(directory / "release")
            config = self.config(manifest, train_roots=2, val_roots=1)
            roots, selection = select_roots(config, directory / "out")
            self.assertEqual(len(roots), 3)
            self.assertEqual(selection["counts"]["train"]["selected"], 2)
            self.assertEqual(selection["counts"]["val"]["selected"], 1)
            self.assertEqual({root["split"] for root in roots}, {"train", "val"})


class JsonlStoreTests(unittest.TestCase):
    def store(self, directory, identity="abc"):
        return JsonlStore(Path(directory) / "records.jsonl", identity)

    def test_append_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.store(directory)
            store.append({"identity_sha256": "abc", "root_id": "a", "value": 1})
            store.append({"identity_sha256": "abc", "root_id": "b", "value": 2})
            records = store.load()
            self.assertEqual(list(records), ["a", "b"])
            self.assertEqual(records["b"]["value"], 2)

    def test_bad_tail_is_truncated_and_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.store(directory)
            store.append({"identity_sha256": "abc", "root_id": "a"})
            committed = store.path.read_bytes()
            store.path.write_bytes(committed + b'{"identity_sha256": "abc"')
            self.assertEqual(list(store.load()), ["a"])
            self.assertTrue(store.path.read_bytes().endswith(b"\n"))
            store.append({"identity_sha256": "abc", "root_id": "b"})
            self.assertEqual(list(store.load()), ["a", "b"])

    def test_malformed_last_complete_line_is_dropped_but_mid_file_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.store(directory)
            store.append({"identity_sha256": "abc", "root_id": "a"})
            store.path.write_bytes(store.path.read_bytes() + b"not json\n")
            self.assertEqual(list(store.load()), ["a"])
            self.assertTrue(store.path.read_bytes().endswith(b"\n"))
            store.path.write_bytes(b"not json\n" + store.path.read_bytes())
            with self.assertRaises(ValueError):
                store.load()

    def test_wrong_identity_duplicate_and_missing_keys_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.store(directory)
            store.append({"identity_sha256": "abc", "root_id": "a"})
            store.path.write_bytes(
                store.path.read_bytes() + b'{"identity_sha256":"other","root_id":"b"}\n'
            )
            with self.assertRaises(ValueError):
                store.load()
        with tempfile.TemporaryDirectory() as directory:
            store = self.store(directory)
            store.append({"identity_sha256": "abc", "root_id": "a"})
            store.path.write_bytes(
                store.path.read_bytes() + b'{"identity_sha256":"abc","root_id":"a"}\n'
            )
            with self.assertRaises(ValueError):
                store.load()
        with tempfile.TemporaryDirectory() as directory:
            store = self.store(directory)
            store.path.write_bytes(b'{"identity_sha256":"abc"}\n')
            with self.assertRaises(ValueError):
                store.load()


class SummaryTests(unittest.TestCase):
    def config(self):
        return PilotConfig(
            release="release.json",
            release_sha256="0" * 64,
            train_roots=1,
            val_roots=1,
            teachers=(TEACHER,),
        )

    def metrics(self, **overrides):
        base = {
            "nodes": 1,
            "terminal_nodes": 0,
            "analysed_nodes": 1,
            "expanded_nodes": 0,
            "non_root_expanded_nodes": 0,
            "max_depth": 0,
            "max_width": 0,
            "root_width": 0,
            "proposed": 0,
            "updates": 0,
            "revisions": 0,
            "leaf_count": 1,
            "leaf_shift": 0,
            "states": 2,
            "state_transitions": 1,
            "state_changes": 0,
            "final_basis": "shallow",
            "final_move": "e2e4",
            "final_value": 0.0,
            "overturned": False,
            "stop_depths": {"0": 1},
            "board_requests": 0,
            "board_response_tokens": 0,
            "no_tool_tokens": 5,
            "tool_tokens": 5,
            "model_tokens": 5,
            "token_increase": 0,
            "engine_calls": 0,
            "analysis_ms": 0,
            "build_ms": 0,
        }
        base.update(overrides)
        return base

    def test_aggregates_rates_and_model_deltas_by_hand(self):
        config = self.config()
        records = [
            (
                "root-a",
                {
                    "analysis": {
                        "calls": 2,
                        "elapsed_ms": 100,
                        "nodes": 3,
                        "terminal_nodes": 0,
                    },
                    "elapsed_ms": 200,
                    "teachers": {
                        TEACHER: {
                            "metrics": self.metrics(
                                nodes=3,
                                non_root_expanded_nodes=1,
                                max_depth=2,
                                max_width=2,
                                proposed=2,
                                updates=2,
                                revisions=1,
                                states=3,
                                state_transitions=2,
                                state_changes=1,
                                final_basis="completed",
                                overturned=True,
                                stop_depths={"2": 1},
                                board_requests=1,
                                board_response_tokens=10,
                                no_tool_tokens=20,
                                tool_tokens=31,
                                model_tokens=21,
                                token_increase=11,
                                engine_calls=2,
                                analysis_ms=100,
                                build_ms=10,
                            )
                        }
                    },
                },
            ),
            (
                "root-b",
                {
                    "analysis": {
                        "calls": 0,
                        "elapsed_ms": 0,
                        "nodes": 1,
                        "terminal_nodes": 0,
                    },
                    "elapsed_ms": 0,
                    "teachers": {TEACHER: {"metrics": self.metrics()}},
                },
            ),
        ]
        roots = [
            {"root_id": "root-a", "split": "train"},
            {"root_id": "root-b", "split": "val"},
        ]
        quality = {
            "root-b": {
                "teachers": {
                    TEACHER: {
                        "top1": True,
                        "delta": 0.1,
                        "reference": {"value": 0.2},
                        "restricted": {"value": 0.3},
                    }
                }
            }
        }
        summary = build_summary(
            config,
            roots,
            dict(records),
            quality,
            {"ok": True},
            [],
            [],
        )
        teacher = summary["teachers"][TEACHER]
        self.assertEqual(teacher["roots"], 2)
        self.assertEqual(
            teacher["nodes"], {"total": 4, "mean": 2.0, "min": 1, "max": 3}
        )
        self.assertEqual(teacher["depth"], {"mean": 1.0, "max": 2})
        self.assertEqual(teacher["width"], {"mean": 1.0, "max": 2})
        self.assertEqual(teacher["root_candidates"], {"mean": 1.0})
        self.assertEqual(
            teacher["tokens"],
            {
                "no_tool_total": 25,
                "no_tool_mean": 12.5,
                "tool_total": 36,
                "tool_mean": 18.0,
                "model_total": 26,
                "model_mean": 13.0,
            },
        )
        self.assertEqual(teacher["engine_calls"], {"total": 2, "mean": 1.0})
        self.assertEqual(teacher["non_root_branch_rate"], 0.5)
        self.assertEqual(
            teacher["first_candidate_overturned"],
            {"roots_with_multiple": 1, "overturned": 1, "rate": 1.0},
        )
        self.assertEqual(
            teacher["evaluation_revision"],
            {"updates": 2, "revisions": 1, "rate": 0.5},
        )
        self.assertEqual(
            teacher["truncation_change"],
            {
                "states": 5,
                "transitions": 3,
                "changes": 1,
                "rate": 1 / 3,
                "roots_with_changes": 1,
                "change_root_rate": 0.5,
            },
        )
        self.assertEqual(teacher["stop_depths"], {"0": 1, "2": 1})
        self.assertEqual(teacher["final_basis"], {"completed": 1, "shallow": 1})
        board = teacher["board"]
        self.assertEqual(board["requests"], 1)
        self.assertEqual(board["request_rate"], 0.5)
        self.assertAlmostEqual(board["response_ratio_no_tool"], 0.4)
        self.assertAlmostEqual(board["response_ratio_tool"], 10 / 36)
        self.assertAlmostEqual(board["increase_rate"], 11 / 25)
        quality_summary = teacher["quality"]
        self.assertEqual(quality_summary["top1"], 1)
        self.assertEqual(quality_summary["top1_rate"], 1.0)
        self.assertAlmostEqual(quality_summary["mean_delta"], 0.1)
        self.assertAlmostEqual(quality_summary["mean_abs_delta"], 0.1)
        self.assertEqual(quality_summary["better"], 1)
        self.assertEqual(summary["synthesis"]["written"], 2)
        self.assertEqual(summary["synthesis"]["missing"], 0)
        self.assertEqual(summary["analysis"]["calls_total"], 2)
        self.assertEqual(summary["analysis"]["mean_nodes"], 2.0)


if __name__ == "__main__":
    unittest.main()
