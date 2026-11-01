import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from pondera.data.pipeline import (
    CollectionError,
    _check_sidecar,
    load_config,
    run_pipeline,
    sample_depth,
    sample_presentation,
    sample_root_settings,
    sample_width,
    validate_release,
)
from pondera.data.trace import SynthesisError, load_sidecar

FIXTURES = Path(__file__).parent / "fixtures" / "trace_v2"


def config_text(depth=None, engine=None):
    text = Path("configs/trace.toml").read_text(encoding="utf-8")
    if depth is not None:
        text = text.replace("depths = 16", f"depths = {depth}")
    if engine is not None:
        text = text.replace('engine = "stockfish"', f'engine = "{engine}"')
    return text


def distributed_text():
    return (
        config_text()
        .replace("widths = 1", "widths = { 1 = 0.25, 3 = 0.75 }")
        .replace("depths = 16", "depths = { 8 = 0.5, 16 = 0.5 }")
        .replace(
            "presentations = { zero = 0.10, full = 0.50, random = 0.40 }",
            "presentations = { zero = 0.2, full = 0.5, random = 0.3 }",
        )
    )


class ConfigTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.toml"
            path.write_text(text, encoding="utf-8")
            return load_config(path)

    def test_default_config_is_the_single_authority(self):
        config = load_config()
        self.assertEqual(config.search.depth, 16)
        self.assertEqual(config.search.teacher_width, 1)
        self.assertEqual(config.search.threads, 1)
        self.assertEqual(config.search.hash_mb, 256)
        self.assertTrue(config.search.fresh_engine_per_root)
        self.assertEqual(config.synthesis.board_descent_hazard, 0.15)
        self.assertEqual(config.synthesis.board_frontier_p, 0.5)
        self.assertEqual(config.synthesis.zero, 0.10)
        self.assertEqual(config.synthesis.full, 0.50)
        self.assertEqual(config.synthesis.random, 0.40)
        self.assertTrue(config.synthesis.claim_draw)
        self.assertEqual(config.roots.train, 1000)
        self.assertEqual(config.seed, 640)
        self.assertEqual(config.release.shard_size, 1024)

    def test_scalar_and_table_distributions_have_one_sampling_rule(self):
        config = self.load(distributed_text())
        roots = [f"root-{index}" for index in range(10000)]
        widths = Counter(sample_width(config.sampling, 640, root, 1) for root in roots)
        depths = Counter(sample_depth(config.sampling, 640, root, 16) for root in roots)
        presentations = Counter(
            sample_presentation(config.sampling, 640, root) for root in roots
        )
        self.assertAlmostEqual(widths[1] / len(roots), 0.25, delta=0.02)
        self.assertAlmostEqual(widths[3] / len(roots), 0.75, delta=0.02)
        self.assertAlmostEqual(depths[8] / len(roots), 0.5, delta=0.02)
        self.assertAlmostEqual(presentations["zero"] / len(roots), 0.2, delta=0.02)
        root = "stable-root"
        self.assertEqual(
            sample_width(config.sampling, 640, root, 1),
            sample_width(config.sampling, 640, root, 1),
        )

    def test_sampling_attaches_and_preserves_settings(self):
        config = self.load(distributed_text())
        roots = [{"root_id": f"root-{index}", "split": "train"} for index in range(200)]
        sampled = sample_root_settings(roots, config, 640)
        self.assertEqual(sampled, sample_root_settings(sampled, config, 640))
        for root in sampled:
            self.assertIn(root["teacher_width"], (1, 3))
            self.assertIn(root["depth"], (8, 16))
            self.assertIn(root["presentation"], ("zero", "full", "random"))
        self.assertNotIn("teacher_width", roots[0])
        changed = sum(
            left["teacher_width"] != right["teacher_width"]
            or left["depth"] != right["depth"]
            or left["presentation"] != right["presentation"]
            for left, right in zip(sampled, sample_root_settings(roots, config, 641))
        )
        self.assertGreater(changed, 0)

    def test_zero_probability_keys_are_never_drawn(self):
        text = config_text().replace("widths = 1", "widths = { 1 = 0.0, 3 = 1.0 }")
        config = self.load(text)
        draws = {
            sample_width(config.sampling, 640, f"root-{index}", 1)
            for index in range(100)
        }
        self.assertEqual(draws, {3})

    def test_rejects_unknown_and_inconsistent_values(self):
        base = config_text()
        cases = [
            base.replace("[paths]", "[paths]\nunknown = 1"),
            base.replace("threads = 1", "threads = 2"),
            base.replace(
                "fresh_engine_per_root = true", "fresh_engine_per_root = false"
            ),
            base.replace("zero = 0.10", "zero = 0.20"),
            base.replace("random_min_fraction = 0.05", "random_min_fraction = 0.99"),
            base.replace("[roots]", "[root]"),
            base.replace("depths = 16", "depths = 0"),
            base.replace("widths = 1", "widths = { 0 = 1.0 }"),
            base.replace(
                "presentations = { zero = 0.10, full = 0.50, random = 0.40 }",
                "presentations = { last_root = 1.0 }",
            ),
        ]
        for text in cases:
            with self.subTest(text=text[:80]):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "trace.toml"
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_config(path)


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
        self.assertEqual(
            _check_sidecar(
                {**root, "teacher_width": 1, "depth": 4},
                sidecar,
                config,
                sidecar.engine,
            ),
            1,
        )
        self.assertEqual(_check_sidecar(root, sidecar, config, sidecar.engine), 1)
        for bad in (
            {**root, "teacher_width": 2, "depth": 4},
            {**root, "teacher_width": 1, "depth": 5},
        ):
            with self.subTest(root=bad):
                with self.assertRaises(SynthesisError):
                    _check_sidecar(bad, sidecar, config, sidecar.engine)
        with self.assertRaises(CollectionError):
            _check_sidecar(root, sidecar, self.config(depth=5), sidecar.engine)

    def test_sidecar_must_match_the_pinned_engine_identity(self):
        fixture = FIXTURES / "start_d4"
        root = json.loads((fixture / "root.json").read_text(encoding="utf-8"))
        sidecar = load_sidecar(fixture / "sidecar.jsonl")
        config = self.config()
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
    def test_engine_produces_a_release_without_a_sidecar_directory(self):
        engine = os.environ.get("PONDERA_TRACE_ENGINE")
        patch = os.environ.get("PONDERA_TRACE_PATCH")
        control = os.environ.get("PONDERA_TRACE_CONTROL")
        if not (engine and patch and control):
            self.skipTest("set PONDERA_TRACE_* to the instrumented engine and patch")
        for path in (engine, patch, control):
            if not Path(path).is_file():
                self.skipTest(f"missing instrumented artifact: {path}")
        fixture = FIXTURES / "start_d4"
        root = json.loads((fixture / "root.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config_path = directory / "trace.toml"
            config_path.write_text(
                config_text(depth=4, engine=str(Path(engine).resolve()))
                .replace(
                    'roots = ".local/data/roots.jsonl"',
                    f'roots = "{directory / "roots.jsonl"}"',
                )
                .replace(
                    'output = ".local/runs/trace"',
                    f'output = "{directory / "release"}"',
                )
                .replace('instrumentation = ""', f'instrumentation = "{patch}"')
                .replace('control = ""', f'control = "{control}"'),
                encoding="utf-8",
            )
            (directory / "roots.jsonl").write_text(
                json.dumps(root) + "\n", encoding="utf-8"
            )
            result = run_pipeline(load_config(config_path), workers=1)
            self.assertEqual(result["roots"], 1)
            self.assertFalse((directory / "release" / "roots").exists())
            self.assertTrue(validate_release(directory / "release", workers=1)["ok"])


if __name__ == "__main__":
    unittest.main()
