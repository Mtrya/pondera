import json
import tempfile
import threading
import tomllib
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from trace_fixtures import read_release_rows, tie_sidecar_lines, trace_config_text

from pondera.chess.symmetry import reflect_uci
from pondera.chess.vocabulary import MOVE, UCI_MOVE_TO_IDX
from pondera.data.config import load_config, parse_recipe
from pondera.data.dataset import (
    DATA_ARRAYS,
    TracePositions,
    load_manifest,
    read_root_rows,
)
from pondera.data.pipeline import (
    load_trace_arrays,
    produce_shards,
    sample_root_settings,
    validate_release,
    validate_release_identity,
    verify_shard_files,
)
from pondera.data.trace import replay

FIXTURES = Path(__file__).parent / "fixtures" / "trace_v2"
CASES = (
    "start_d4",
    "middlegame_black_d4",
    "mate_in_1_d4",
    "claim_draw_d4",
    "stalemate_d4",
)
W3_CASES = (
    "middlegame_black_w3",
    "mate_in_1_w3",
    "two_moves_w3",
)
MIXED_RECIPE = """
[recipe]
joint = [{width = 1, depth = 4, probability = 0.5}, {width = 3, depth = 4, probability = 0.5}]
presentations = {zero = 0.2, full = 0.5, random = 0.3}
"""


def config_text(width=1, recipe_text=None):
    recipe = tomllib.loads(recipe_text)["recipe"] if recipe_text is not None else {}
    return trace_config_text(width=width, **recipe)


def build_run(
    directory, cases=CASES, width=1, sidecar_mutator=None, recipe_text=None, seed=640
):
    directory = Path(directory)
    config_path = directory / "trace.toml"
    config_path.write_text(
        config_text(width, recipe_text).replace("seed = 640", f"seed = {seed}"),
        encoding="utf-8",
    )
    config = load_config(config_path)
    run = directory / "run"
    run.mkdir(parents=True)
    roots = []
    provider = {}
    for case in cases:
        fixture = FIXTURES / case
        root = json.loads((fixture / "root.json").read_text(encoding="utf-8"))
        lines = (fixture / "sidecar.jsonl").read_text(encoding="utf-8").splitlines()
        if sidecar_mutator is not None:
            lines = sidecar_mutator(case, lines)
        provider[root["root_id"]] = lines
        roots.append(root)
    roots = sample_root_settings(roots, config, seed)
    return config, roots, run, provider


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config, self.roots, self.run, self.provider = build_run(self.root)
        base = self.run
        self.result = produce_shards(
            self.roots,
            base / "release",
            self.config,
            sidecar_for_root=self.provider,
            shard_size=2,
            workers=1,
        )
        self.release = base / "release"

    def test_release_layout_identity_and_validation(self):
        self.assertEqual(self.result["roots"], len(CASES))
        # One presentation per root: five roots produce 2 + 2 + 1 rows.
        self.assertEqual(len(self.result["shards"]), 3)
        self.assertEqual([shard["rows"] for shard in self.result["shards"]], [2, 2, 1])
        release = load_manifest(self.release / "release.json")
        self.assertEqual(release["format_version"], 10)
        self.assertEqual(release["grammar_version"], 2)
        self.assertEqual(release["teacher"]["schema_version"], 3)
        self.assertEqual(release["search"], {"hash_mb": 256})
        self.assertEqual(release["recipe"], self.config.sampling.identity())
        self.assertEqual(release["engine"]["name"], "stockfish")
        self.assertEqual(release["presentation"], self.config.presentation_dict())
        self.assertEqual(release["metadata"], validate_release_identity(release))
        validation = validate_release(self.release, self.roots, workers=1)
        self.assertTrue(validation["ok"])
        self.assertEqual(validation["roots"], len(CASES))
        self.assertEqual(validation["rows"], len(CASES))
        self.assertEqual(validation["counts"]["train"], len(CASES))
        self.assertTrue((self.release / "validation.json").exists())

    def test_hf_columns_preserve_the_shared_scalar_contract(self):
        expected = {
            "fen": pa.string(),
            "presentation": pa.string(),
            "repetition": pa.int16(),
            "turn": pa.string(),
            "teacher_depth": pa.int16(),
            "teacher_width": pa.int16(),
            "trace_text": pa.string(),
            "final_move": pa.string(),
            "final_value": pa.float32(),
            "mate_distance": pa.int16(),
            "source": pa.string(),
        }
        for shard in self.result["shards"]:
            table = pq.read_table(self.release / shard["path"] / "data.parquet")
            self.assertEqual(len(table.schema), 17)
            for name, dtype in expected.items():
                self.assertEqual(table.schema.field(name).type, dtype)
            for row in table.to_pylist():
                self.assertEqual(len(row["trace_token_ids"]), row["trace_len"])
                self.assertEqual(len(row["state_best"]), len(row["state_value"]))
                self.assertTrue(row["trace_text"].endswith(" END"))

    def test_parallel_validation_matches_sequential(self):
        sequential = validate_release(self.release, self.roots, workers=1)
        parallel = validate_release(self.release, self.roots, workers=4)
        self.assertEqual(parallel, sequential)

    def expected_shard_metadata(self, release):
        expected = {
            key: release[key]
            for key in (
                "format_version",
                "grammar_version",
                "tokenizer_version",
                "role_count",
            )
        }
        expected.update(release["metadata"])
        return expected

    def test_shards_have_arrays_champions_and_identical_suffixes(self):
        release = load_manifest(self.release / "release.json")
        expected = self.expected_shard_metadata(release)
        suffixes = {}
        for shard in release["shards"]:
            manifest, rows, arrays = verify_shard_files(self.release, shard, expected)
            self.assertEqual(manifest["rows"], shard["rows"])
            champions = {
                row["root_id"]: row
                for row in (
                    json.loads(line)
                    for line in (self.release / shard["path"] / "champions.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                )
            }
            for index, row in enumerate(rows):
                length = int(arrays["trace_len"][index])
                tokens = arrays["trace_token_ids"][index][:length]
                program = replay(
                    tokens,
                    row["fen"],
                    row["repetition"],
                    teacher_width=1,
                )
                final = next(
                    step for step in program.steps if step.operation == "final"
                )
                suffix = tuple(int(token) for token in tokens[final.start :])
                key = row["root_id"]
                self.assertEqual(suffixes.setdefault(key, suffix), suffix)
                self.assertEqual(len(champions[key]["rounds"]), row["teacher_depth"])
                thought = [
                    step
                    for step in program.steps
                    if step.operation
                    not in {"final", "final_evaluate", "final_pv", "end"}
                ]
                self.assertEqual(int(arrays["state_count"][index]), len(thought))
        self.assertEqual(set(suffixes), {root["root_id"] for root in self.roots})
        positions = TracePositions(self.release, "train")
        self.assertEqual(len(positions), len(CASES))
        row = positions[0]
        self.assertGreater(int(row[2]), 0)

    def test_resume_and_shard_recovery(self):

        again = produce_shards(
            self.roots,
            self.release,
            self.config,
            sidecar_for_root=self.provider,
            shard_size=2,
            workers=1,
        )
        self.assertEqual(again["roots"], len(CASES))
        self.assertEqual(len(again["shards"]), 3)
        self.assertEqual(again["statistics"], self.result["statistics"])
        (self.release / "release.json").unlink()
        recovered = produce_shards(
            self.roots,
            self.release,
            self.config,
            sidecar_for_root=self.provider,
            shard_size=2,
            workers=1,
        )
        self.assertEqual(recovered["roots"], len(CASES))
        self.assertEqual(recovered["statistics"], self.result["statistics"])
        self.assertTrue(validate_release(self.release, self.roots, workers=1)["ok"])

    def test_resume_rejects_a_different_shard_layout(self):
        with self.assertRaisesRegex(ValueError, "different shard layout"):
            produce_shards(
                self.roots,
                self.release,
                self.config,
                sidecar_for_root=self.provider,
                shard_size=3,
                workers=1,
            )

    def test_resume_rejects_a_changed_joint_recipe(self):
        recipe = {
            **self.config.sampling.identity(),
            "joint": [
                {"width": 1, "depth": 4, "probability": 0.5},
                {"width": 3, "depth": 4, "probability": 0.5},
            ],
        }
        changed = replace(self.config, sampling=parse_recipe(recipe))
        with self.assertRaises(ValueError):
            produce_shards(
                self.roots,
                self.release,
                changed,
                sidecar_for_root=self.provider,
                shard_size=2,
                workers=1,
            )

    def test_validation_rejects_a_wrong_trace_length(self):
        path = self.release / "shard-000000" / "data.parquet"
        table = pq.read_table(path)
        rows = table.to_pylist()
        rows[0]["trace_len"] += 1
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        with self.assertRaises(ValueError):
            validate_release(self.release, workers=1)

    def test_disjoint_jobs_merge_their_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            config, roots, run, provider = build_run(
                directory, cases=("start_d4", "middlegame_black_d4")
            )
            barrier = threading.Barrier(2)

            def synchronized(root):
                barrier.wait(timeout=10)
                return provider[root["root_id"]]

            release = run / "release"
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(
                        produce_shards,
                        roots,
                        release,
                        config,
                        sidecar_for_root=synchronized,
                        shard_size=1,
                        workers=1,
                        start_shard=start,
                        end_shard=start + 1,
                    )
                    for start in (0, 1)
                ]
                for future in futures:
                    future.result()
            manifest = load_manifest(release / "release.json")
            self.assertEqual(
                [shard["path"] for shard in manifest["shards"]],
                ["shard-000000", "shard-000001"],
            )
            self.assertTrue(validate_release(release, roots, workers=1)["ok"])

    def test_claim_root_is_materialized_without_a_variation(self):
        release = load_manifest(self.release / "release.json")
        claim_roots = set()
        expected = self.expected_shard_metadata(release)
        for shard in release["shards"]:
            _, rows, _ = verify_shard_files(self.release, shard, expected)
            for row in rows:
                if row["final_move"] == "<claim_draw>":
                    claim_roots.add(row["root_id"])
                    self.assertEqual(row["mate_distance"], None)
        claim = json.loads((FIXTURES / "claim_draw_d4" / "root.json").read_text())
        self.assertEqual(claim_roots, {claim["root_id"]})


class WideReleaseTests(unittest.TestCase):
    def test_wide_teacher_release_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            config, roots, run, provider = build_run(directory, cases=W3_CASES, width=3)
            base = run
            produce_shards(
                roots,
                base / "release",
                config,
                sidecar_for_root=provider,
                shard_size=1,
                workers=1,
            )
            release = load_manifest(base / "release" / "release.json")
            self.assertEqual(release["teacher"]["schema_version"], 4)
            self.assertEqual(release["recipe"], config.sampling.identity())
            self.assertEqual(release["engine"]["name"], "stockfish")
            validation = validate_release(base / "release", roots, workers=1)
            self.assertTrue(validation["ok"])
            self.assertEqual(validation["rows"], len(W3_CASES))

    def test_mixed_width_recipe_release_and_validation(self):
        cases = ("start_d4", "middlegame_black_w3", "two_moves_w3")
        # Seed 10 assigns the widths (1, 3, 3) of these real sidecars.
        with tempfile.TemporaryDirectory() as directory:
            config, roots, run, provider = build_run(
                directory,
                cases=cases,
                width=1,
                recipe_text=MIXED_RECIPE,
                seed=10,
            )
            self.assertEqual(
                {root["root_id"]: root["teacher_width"] for root in roots},
                {
                    "bc6ae1d761cf700dfb66ef956ac17bef4a90e0f4846624e57d22ba65f92426c7": 1,
                    "23fd2fbf6d032131267d358c35ba053a17fecb87facfabf0b3331e01e0bb8c55": 3,
                    "6c9691cab19e3988acbe741a28cf1739fe971be60fdba32fe431f2e43c89a59a": 3,
                },
            )
            base = run
            produce_shards(
                roots,
                base / "release",
                config,
                sidecar_for_root=provider,
                shard_size=2,
                workers=1,
            )
            release = load_manifest(base / "release" / "release.json")
            self.assertEqual(release["teacher"]["schema_version"], 4)
            self.assertEqual(release["recipe"], config.sampling.identity())
            validation = validate_release(base / "release", roots, workers=1)
            self.assertTrue(validation["ok"])
            self.assertEqual(validation["rows"], len(cases))
            records = read_release_rows(base / "release")
            self.assertEqual(
                {
                    row["root_id"]: row["presentation"]
                    for row in (record["row"] for record in records.values())
                },
                {root["root_id"]: root["presentation"] for root in roots},
            )
            self.assertEqual(
                {
                    row["root_id"]: row["teacher_width"]
                    for row in (record["row"] for record in records.values())
                },
                {root["root_id"]: root["teacher_width"] for root in roots},
            )
            capped = next(
                root for root in roots if root["root_id"].startswith("6c9691")
            )
            self.assertEqual(chess.Board(capped["fen"]).legal_moves.count(), 2)
            stored = records[capped["root_id"], capped["presentation"]]["row"]
            self.assertEqual(stored["teacher_width"], 3)
            self.assertNotIn("effective_width", stored)

    def test_single_width_distribution_uses_schema_three(self):
        recipe = """
[recipe]
joint = [{width = 1, depth = 4, probability = 1.0}]
"""
        with tempfile.TemporaryDirectory() as directory:
            config, roots, run, provider = build_run(
                directory, cases=("start_d4",), width=3, recipe_text=recipe
            )
            self.assertEqual({root["teacher_width"] for root in roots}, {1})
            base = run
            produce_shards(
                roots,
                base / "release",
                config,
                sidecar_for_root=provider,
                shard_size=1,
                workers=1,
            )
            release = load_manifest(base / "release" / "release.json")
            self.assertEqual(release["teacher"]["schema_version"], 3)
            validation = validate_release(base / "release", roots, workers=1)
            self.assertTrue(validation["ok"])

    def test_validation_reconstructs_joint_settings(self):
        for width, depth in ((3, 4), (1, 5)):
            with (
                self.subTest(width=width, depth=depth),
                tempfile.TemporaryDirectory() as directory,
            ):
                config, roots, run, provider = build_run(directory, cases=("start_d4",))
                release_dir = run / "release"
                produce_shards(
                    roots, release_dir, config, sidecar_for_root=provider, workers=1
                )
                release_path = release_dir / "release.json"
                release = load_manifest(release_path)
                recipe = {
                    **release["recipe"],
                    "joint": [
                        {"width": 1, "depth": 4, "probability": 0.0},
                        {"width": width, "depth": depth, "probability": 1.0},
                    ],
                }
                release["recipe"] = release["metadata"]["recipe"] = recipe
                release_path.write_text(json.dumps(release))
                for shard in release["shards"]:
                    path = release_dir / shard["path"] / "manifest.json"
                    manifest = load_manifest(path)
                    manifest["recipe"] = recipe
                    path.write_text(json.dumps(manifest))
                validate_release_identity(release)
                with self.assertRaises(ValueError):
                    validate_release(release_dir, workers=1)

    def test_presentation_sampling_is_deterministic_on_resume(self):
        cases = ("start_d4", "middlegame_black_w3", "two_moves_w3")
        with tempfile.TemporaryDirectory() as directory:
            config, roots, run, provider = build_run(
                directory, cases=cases, recipe_text=MIXED_RECIPE, seed=10
            )
            produce_shards(
                roots,
                run / "release",
                config,
                sidecar_for_root=provider,
                shard_size=2,
                workers=1,
            )
            first = read_release_rows(run / "release")
            produce_shards(
                roots,
                run / "release",
                config,
                sidecar_for_root=provider,
                shard_size=2,
                workers=1,
            )
            second = read_release_rows(run / "release")
            self.assertEqual(first, second)
            self.assertEqual(len(second), len(cases))
            # Probabilities are part of identity even when these roots draw
            # the same settings under both distributions.
            recipe = config.sampling.identity()
            recipe["joint"][0]["probability"] = 0.51
            recipe["joint"][1]["probability"] = 0.49
            changed = replace(config, sampling=parse_recipe(recipe))
            self.assertEqual(roots, sample_root_settings(roots, changed, changed.seed))
            with self.assertRaises(ValueError):
                produce_shards(
                    roots,
                    run / "release",
                    changed,
                    sidecar_for_root=provider,
                    shard_size=2,
                    workers=1,
                )

    def test_tied_wide_round_releases_with_first_come_state(self):
        with tempfile.TemporaryDirectory() as directory:
            config, roots, run, provider = build_run(
                directory,
                cases=("middlegame_black_w3",),
                width=3,
                sidecar_mutator=tie_sidecar_lines,
                recipe_text="[recipe]\npresentations = {full = 1.0}",
            )
            base = run
            produce_shards(
                roots,
                base / "release",
                config,
                sidecar_for_root=provider,
                shard_size=1,
                workers=1,
            )
            validation = validate_release(base / "release", roots, workers=1)
            self.assertTrue(validation["ok"])
            shard = base / "release" / "shard-000000"
            arrays = load_trace_arrays(shard, DATA_ARRAYS)
            self.assertEqual(arrays["trace_token_ids"].dtype, np.int32)
            self.assertEqual(arrays["trace_len"].dtype, np.int32)
            self.assertEqual(arrays["state_count"].dtype, np.int32)
            for name in ("state_best", "state_value"):
                self.assertEqual(arrays[name].dtype, np.int32)
            rows = read_root_rows(shard)
            full = next(
                index for index, row in enumerate(rows) if row["presentation"] == "full"
            )
            tokens = arrays["trace_token_ids"][full]
            program = replay(tokens, rows[full]["fen"], rows[full]["repetition"], 3)
            depths = [
                index
                for index, step in enumerate(program.steps)
                if step.operation == "depth"
            ]
            best = arrays["state_best"][full]
            kept = MOVE + UCI_MOVE_TO_IDX[reflect_uci("f6e4")]
            self.assertEqual(int(best[depths[1] - 1]), kept)


if __name__ == "__main__":
    unittest.main()
