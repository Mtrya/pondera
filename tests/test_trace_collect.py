import json
import os
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

import chess
from trace_fixtures import read_release_rows
from trace_fixtures import trace_config_text as config_text

from pondera.data.pipeline import (
    CollectionError,
    _check_sidecar,
    load_config,
    parse_recipe,
    produce_shards,
    run_pipeline,
    sample_root_settings,
    validate_release,
)
from pondera.data.roots import root_id_for
from pondera.data.trace import SynthesisError, load_sidecar

FIXTURES = Path(__file__).parent / "fixtures" / "trace_v2"


class RecipeTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.toml"
            path.write_text(text, encoding="utf-8")
            return load_config(path)

    def test_root_settings_are_recomputed_from_the_active_recipe(self):
        roots = [
            {"root_id": f"root-{index}", "fen": chess.STARTING_FEN}
            for index in range(200)
        ]
        first = self.load(config_text())
        sampled = sample_root_settings(roots, first, first.seed)
        self.assertEqual(sampled, sample_root_settings(sampled, first, first.seed))

        changed = self.load(config_text(width=3, depth=8))
        changed = sample_root_settings(roots, changed, changed.seed)
        self.assertEqual({root["teacher_width"] for root in changed}, {3})
        self.assertEqual({root["depth"] for root in changed}, {8})
        self.assertEqual(
            [root["presentation"] for root in sampled],
            [root["presentation"] for root in changed],
        )

    def test_joint_support_probabilities_and_stable_assignments(self):
        joint = [
            {"width": 1, "depth": 8, "probability": 0.25},
            {"width": 3, "depth": 4, "probability": 0.75},
        ]
        config = self.load(config_text(joint=joint))
        roots = [
            {"root_id": f"root-{i}", "fen": chess.STARTING_FEN} for i in range(2000)
        ]
        sampled = sample_root_settings(roots, config, 640)
        counts = Counter((r["teacher_width"], r["depth"]) for r in sampled)
        self.assertEqual(set(counts), {(1, 8), (3, 4)})
        self.assertAlmostEqual(counts[1, 8] / len(roots), 0.25, delta=0.04)
        expected = {r["root_id"]: r for r in sampled}
        self.assertEqual(
            expected,
            {
                r["root_id"]: r
                for r in sample_root_settings(reversed(roots), config, 640)
            },
        )
        chunked = [
            r
            for start in range(0, len(roots), 17)
            for r in sample_root_settings(roots[start : start + 17], config, 640)
        ]
        self.assertEqual(sampled, chunked)
        reordered = self.load(config_text(joint=list(reversed(joint))))
        self.assertEqual(config.sampling, reordered.sampling)
        self.assertEqual(sampled, sample_root_settings(sampled, reordered, 640))
        self.assertEqual(config.sampling, parse_recipe(config.sampling.identity()))

    def test_full_width_resolves_legal_root_moves(self):
        roots = [
            json.loads((FIXTURES / name / "root.json").read_text())
            for name in ("start_d4", "two_moves_w3")
        ]
        roots.append(
            {"root_id": "one-legal-move", "fen": "1r6/8/8/8/8/8/2k5/K7 w - - 0 1"}
        )
        config = self.load(config_text(width="full"))
        sampled = sample_root_settings(roots, config, 640)
        self.assertEqual([r["teacher_width"] for r in sampled], [20, 2, 1])
        for root in sampled:
            self.assertEqual(
                root["teacher_width"], chess.Board(root["fen"]).legal_moves.count()
            )
            self.assertEqual(root["effective_width"], root["teacher_width"])
        capped = sample_root_settings(roots, self.load(config_text(width=3)), 640)
        self.assertEqual([r["teacher_width"] for r in capped], [3, 3, 3])
        self.assertEqual([r["effective_width"] for r in capped], [3, 2, 1])

    def test_invalid_joint_recipes_are_rejected(self):
        valid = {"width": 1, "depth": 4, "probability": 1.0}
        bad = [
            [],
            {},
            [valid, valid],
            [{"width": 1, "depth": 4}],
            [{**valid, "extra": 0}],
        ]
        for key, values in {
            "width": [0, -1, True, 1.5, "1", "all", None],
            "depth": [0, -1, True, 4.5, "4", None],
            "probability": [-0.1, 0, 0.9, 1.1, True, float("nan"), float("inf")],
        }.items():
            bad.extend([{**valid, key: value}] for value in values)
        for joint in bad:
            with self.subTest(joint=joint):
                with self.assertRaises(ValueError):
                    parse_recipe({"joint": joint, "presentations": {"full": 1.0}})
        for recipe in (
            {"joint": [valid]},
            {
                "widths": {"1": 1.0},
                "depths": {"4": 1.0},
                "presentations": {"full": 1.0},
            },
        ):
            with self.subTest(recipe=recipe):
                with self.assertRaises(ValueError):
                    parse_recipe(recipe)


class SidecarValidationTests(unittest.TestCase):
    def config(self, depth=4):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.toml"
            path.write_text(config_text(depth=depth), encoding="utf-8")
            return load_config(path)

    def test_sidecar_must_match_the_root_teacher_settings(self):
        fixture = FIXTURES / "start_d4"
        root = json.loads((fixture / "root.json").read_text(encoding="utf-8"))
        sidecar = load_sidecar(fixture / "sidecar.jsonl")
        config = self.config()
        root = sample_root_settings([root], config, config.seed)[0]
        self.assertEqual(
            _check_sidecar(
                {**root, "teacher_width": 1, "depth": 4},
                sidecar,
                config,
                sidecar.engine,
            ),
            1,
        )
        for bad in (
            {**root, "teacher_width": 2, "depth": 4},
            {**root, "teacher_width": 1, "depth": 5},
        ):
            with self.subTest(root=bad):
                with self.assertRaises(SynthesisError):
                    _check_sidecar(bad, sidecar, config, sidecar.engine)

    def test_sidecar_must_match_the_engine_name_and_mate_encoding(self):
        fixture = FIXTURES / "start_d4"
        root = json.loads((fixture / "root.json").read_text(encoding="utf-8"))
        sidecar = load_sidecar(fixture / "sidecar.jsonl")
        config = self.config()
        root = sample_root_settings([root], config, config.seed)[0]
        self.assertEqual(
            _check_sidecar(root, sidecar, config, "Stockfish 17.1", (32000, 246)),
            1,
        )
        for engine, teacher in (
            ("stockfish", (32000, 246)),
            ("Stockfish 17.1", (32000, 245)),
        ):
            with self.subTest(engine=engine, teacher=teacher):
                with self.assertRaises(CollectionError):
                    _check_sidecar(root, sidecar, config, engine, teacher)


class DirectPipelineIntegrationTests(unittest.TestCase):
    def engine(self):
        engine = os.environ.get("PONDERA_TRACE_ENGINE")
        if not engine or not Path(engine).is_file():
            self.skipTest("set PONDERA_TRACE_ENGINE to the instrumented engine")
        return Path(engine).resolve()

    def test_engine_produces_a_release_without_a_sidecar_directory(self):
        engine = self.engine()
        fixture = FIXTURES / "start_d4"
        root = json.loads((fixture / "root.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config_path = directory / "trace.toml"
            config_path.write_text(
                config_text(engine=str(engine.resolve()))
                .replace(
                    'roots = ".local/data/roots.jsonl"',
                    f'roots = "{directory / "roots.jsonl"}"',
                )
                .replace(
                    'output = ".local/runs/trace"',
                    f'output = "{directory / "release"}"',
                ),
                encoding="utf-8",
            )
            (directory / "roots.jsonl").write_text(
                json.dumps(root) + "\n", encoding="utf-8"
            )
            result = run_pipeline(load_config(config_path), workers=1)
            self.assertEqual(result["roots"], 1)
            self.assertFalse((directory / "release" / "roots").exists())
            self.assertTrue(validate_release(directory / "release", workers=1)["ok"])

    def test_joint_release_is_stable_across_workers_shards_and_recovery(self):
        engine = self.engine()
        roots = [
            json.loads((FIXTURES / name / "root.json").read_text())
            for name in (
                "start_d4",
                "middlegame_black_d4",
                "mate_in_1_d4",
                "claim_draw_d4",
                "two_moves_w3",
            )
        ]
        fen = "1r6/8/8/8/8/8/2k5/K7 w - - 0 1"
        roots.append(
            {
                "root_id": root_id_for(fen, 1),
                "split": "val",
                "fen": fen,
                "repetition": 1,
                "turn": "white",
                "position_key": " ".join(fen.split()[:4]),
            }
        )
        joint = [
            {"width": 1, "depth": 4, "probability": 0.25},
            {"width": "full", "depth": 3, "probability": 0.75},
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "trace.toml"
            path.write_text(config_text(joint=joint, engine=engine))
            config = load_config(path)
            sampled = sample_root_settings(roots, config, config.seed)
            self.assertEqual({r["depth"] for r in sampled}, {3, 4})
            self.assertIn(1, {r["teacher_width"] for r in sampled})
            self.assertIn(2, {r["teacher_width"] for r in sampled})
            produce_shards(roots, directory / "serial", config, workers=1, shard_size=2)
            produce_shards(
                list(reversed(roots)),
                directory / "parallel",
                config,
                workers=2,
                shard_size=3,
            )
            partial = produce_shards(
                roots,
                directory / "resumed",
                config,
                workers=2,
                shard_size=2,
                end_shard=1,
            )
            self.assertEqual(partial["pending"], 4)
            # Recover a real published shard whose registration was interrupted.
            (directory / "resumed" / "release.json").unlink()
            produce_shards(
                roots, directory / "resumed", config, workers=2, shard_size=2
            )
            expected = read_release_rows(directory / "serial")
            for name in ("serial", "parallel", "resumed"):
                self.assertEqual(expected, read_release_rows(directory / name))
                result = validate_release(directory / name, workers=2)
                self.assertTrue(result["ok"])
                self.assertEqual(result["counts"]["val"], 1)
            full = replace(
                config, sampling=replace(config.sampling, joint=((("full", 3), 1.0),))
            )
            produce_shards(roots[-1:], directory / "full-one", full, workers=1)
            self.assertTrue(validate_release(directory / "full-one", workers=1)["ok"])
            self.assertEqual(
                next(iter(read_release_rows(directory / "full-one").values()))["row"][
                    "teacher_width"
                ],
                1,
            )


if __name__ == "__main__":
    unittest.main()
