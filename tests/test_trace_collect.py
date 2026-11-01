import json
import os
import re
import tempfile
import unittest
from pathlib import Path

from pondera.data.pipeline import (
    CollectionError,
    _check_sidecar,
    load_config,
    run_pipeline,
    sample_root_settings,
    validate_release,
)
from pondera.data.trace import SynthesisError, load_sidecar

FIXTURES = Path(__file__).parent / "fixtures" / "trace_v2"


def config_text(width=1, depth=4, engine=None):
    text = Path("configs/trace.toml").read_text(encoding="utf-8")
    text = re.sub(r"^widths =.*$", f"widths = {width}", text, flags=re.MULTILINE)
    text = re.sub(r"^depths =.*$", f"depths = {depth}", text, flags=re.MULTILINE)
    if engine is not None:
        text = re.sub(
            r'^engine = ".*"$',
            f'engine = "{engine}"',
            text,
            flags=re.MULTILINE,
        )
    return text


class RecipeTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.toml"
            path.write_text(text, encoding="utf-8")
            return load_config(path)

    def test_root_settings_are_recomputed_from_the_active_recipe(self):
        roots = [{"root_id": f"root-{index}", "split": "train"} for index in range(200)]
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

    def test_sidecar_must_match_the_engine_name_and_mate_encoding(self):
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
        if not engine:
            self.skipTest("set PONDERA_TRACE_ENGINE to the instrumented engine")
        engine = Path(engine)
        if not engine.is_file():
            self.skipTest(f"missing instrumented engine: {engine}")
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


if __name__ == "__main__":
    unittest.main()
