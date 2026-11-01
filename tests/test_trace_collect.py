import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from pondera.data.storage import content_sha256, sha256
from pondera.data.trace_collect import (
    CollectionError,
    _load_roots,
    apply_recipe,
    collect,
    select_roots,
)
from pondera.data.trace_config import (
    load_config,
    sample_depth,
    sample_presentation,
    sample_width,
)

FIXTURES = Path(__file__).parent / "fixtures" / "trace_v2"


def config_text(depth=None, engine=None):
    text = Path("configs/trace_v2.toml").read_text(encoding="utf-8")
    if depth is not None:
        text = text.replace("depth = 16", f"depth = {depth}")
    if engine is not None:
        text = text.replace('path = "stockfish"', f'path = "{engine}"')
    return text


RECIPE_TEXT = """
[recipe]
widths = {1 = 0.25, 3 = 0.75}
depths = {8 = 0.5, 16 = 0.5}
presentations = {zero = 0.2, full = 0.5, random = 0.3}
"""


class RecipeConfigTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.toml"
            path.write_text(text, encoding="utf-8")
            return load_config(path)

    def test_default_config_has_no_recipe(self):
        self.assertIsNone(load_config().recipe)

    def test_recipe_distributions_are_sampled_deterministically(self):
        config = self.load(config_text() + RECIPE_TEXT)
        self.assertEqual(
            config.recipe.identity(),
            {
                "widths": {"1": 0.25, "3": 0.75},
                "depths": {"8": 0.5, "16": 0.5},
                "presentations": {"zero": 0.2, "full": 0.5, "random": 0.3},
            },
        )
        self.assertEqual(config.recipe.widest(config.search.teacher_width), 3)
        roots = [f"root-{index}" for index in range(10000)]
        widths = Counter(sample_width(config.recipe, 640, root, 1) for root in roots)
        depths = Counter(sample_depth(config.recipe, 640, root, 16) for root in roots)
        presentations = Counter(
            sample_presentation(config.recipe, 640, root) for root in roots
        )
        self.assertAlmostEqual(widths[1] / len(roots), 0.25, delta=0.02)
        self.assertAlmostEqual(widths[3] / len(roots), 0.75, delta=0.02)
        self.assertAlmostEqual(depths[8] / len(roots), 0.5, delta=0.02)
        self.assertAlmostEqual(depths[16] / len(roots), 0.5, delta=0.02)
        self.assertAlmostEqual(presentations["zero"] / len(roots), 0.2, delta=0.02)
        self.assertAlmostEqual(presentations["full"] / len(roots), 0.5, delta=0.02)
        self.assertAlmostEqual(presentations["random"] / len(roots), 0.3, delta=0.02)
        root = "stable-root"
        self.assertEqual(
            sample_width(config.recipe, 640, root, 1),
            sample_width(config.recipe, 640, root, 1),
        )
        changed = sum(
            sample_width(config.recipe, 1, key, 1)
            != sample_width(config.recipe, 2, key, 1)
            for key in roots[:64]
        )
        self.assertGreater(changed, 0)

    def test_omitted_recipe_tables_fall_back_to_search_settings(self):
        config = self.load(config_text() + "\n[recipe]\nwidths = {3 = 1.0}\n")
        self.assertEqual(config.recipe.identity(), {"widths": {"3": 1.0}})
        self.assertEqual(sample_width(config.recipe, 640, "root", 1), 3)
        self.assertEqual(sample_depth(config.recipe, 640, "root", 16), 16)
        self.assertIsNone(sample_presentation(config.recipe, 640, "root"))
        self.assertFalse(config.recipe.samples_presentations())

    def test_zero_probability_keys_are_never_drawn(self):
        config = self.load(config_text() + "\n[recipe]\nwidths = {1 = 0.0, 3 = 1.0}\n")
        draws = {
            sample_width(config.recipe, 640, f"root-{index}", 1) for index in range(100)
        }
        self.assertEqual(draws, {3})

    def test_recipe_attaches_sampled_teacher_fields_per_root(self):
        config = self.load(config_text() + RECIPE_TEXT)
        roots = [{"root_id": f"root-{index}", "split": "train"} for index in range(200)]
        sampled = apply_recipe(roots, config, 640)
        self.assertEqual(len(sampled), len(roots))
        for root in sampled:
            self.assertIn(root["teacher_width"], (1, 3))
            self.assertIn(root["depth"], (8, 16))
        self.assertEqual(sampled, apply_recipe(roots, config, 640))
        changed = sum(
            left["teacher_width"] != right["teacher_width"]
            or left["depth"] != right["depth"]
            for left, right in zip(sampled, apply_recipe(roots, config, 641))
        )
        self.assertGreater(changed, 0)
        self.assertNotIn("teacher_width", roots[0])

    def test_plain_config_leaves_roots_untouched(self):
        config = load_config()
        roots = [{"root_id": "root", "split": "train"}]
        self.assertEqual(apply_recipe(roots, config, 640), roots)

    def test_existing_root_samples_win_on_resume(self):
        existing = {
            "root_id": "root",
            "split": "train",
            "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
            "repetition": 1,
            "turn": "white",
            "position_key": "8/8/8/8/8/8/8/K6k w - -",
            "teacher_width": 3,
            "depth": 8,
        }
        fresh = {
            key: value
            for key, value in existing.items()
            if key not in {"teacher_width", "depth"}
        }
        fresh.update({"teacher_width": 1, "depth": 16})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "roots.jsonl").write_text(
                json.dumps(existing) + "\n", encoding="utf-8"
            )
            identity, loaded = _load_roots(output, {"seed": 640}, [fresh])
            self.assertEqual(loaded, [existing])
            self.assertEqual(identity["sha256"], content_sha256([existing]))
            fresh["fen"] = "8/8/8/8/8/8/8/7K b - - 0 1"
            with self.assertRaises(CollectionError):
                _load_roots(output, {"seed": 640}, [fresh])

    def test_rejects_invalid_recipes(self):
        base = config_text()
        cases = [
            base + "\n[recipe]\n",
            base + "\n[recipe]\nfoo = 1\n",
            base + "\n[recipe]\nwidths = {1 = 0.5}\n",
            base + "\n[recipe]\nwidths = {1 = -0.1, 3 = 1.1}\n",
            base + "\n[recipe]\nwidths = {0 = 1.0}\n",
            base + "\n[recipe]\nwidths = {1 = 0.5, 3 = 0.5}\ndepths = {16 = 0.5}\n",
            base + "\n[recipe]\npresentations = {last_root = 1.0}\n",
            base + "\n[recipe]\npresentations = {foo = 1.0}\n",
            base + "\n[recpe]\n",
        ]
        for text in cases:
            with self.subTest(recipe=text.rsplit("[recipe]", 1)[-1]):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "trace.toml"
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_config(path)


class ConfigTests(unittest.TestCase):
    def test_the_frozen_config_is_the_authority(self):
        config = load_config()
        self.assertEqual(config.search.depth, 16)
        self.assertEqual(config.search.teacher_width, 1)
        self.assertEqual(config.search.threads, 1)
        self.assertEqual(config.search.hash_mb, 256)
        self.assertTrue(config.search.fresh_engine_per_root)
        self.assertEqual(config.board.board_descent_hazard, 0.15)
        self.assertEqual(config.board.board_frontier_p, 0.5)
        self.assertEqual(config.presentation.zero, 0.10)
        self.assertEqual(config.presentation.full, 0.50)
        self.assertEqual(config.presentation.random, 0.30)
        self.assertEqual(config.presentation.last_root, 0.10)
        self.assertEqual(config.synthesis.confirmatory_repetition, "faithful")
        self.assertTrue(config.synthesis.claim_draw)
        self.assertEqual(config.smoke.roots, 1000)
        self.assertEqual(config.smoke.seed, 640)

    def check_rejected(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.toml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_rejects_unknown_and_inconsistent_values(self):
        text = config_text()
        self.check_rejected(text.replace("[engine]", "[engine]\nunknown = 1"))
        self.check_rejected(text.replace("teacher_width = 1", 'teacher_width = "a"'))
        self.check_rejected(text.replace("threads = 1", "threads = 2"))
        self.check_rejected(
            text.replace(
                "fresh_engine_per_root = true", "fresh_engine_per_root = false"
            )
        )
        self.check_rejected(text.replace("zero = 0.10", "zero = 0.20"))
        self.check_rejected(
            text.replace("random_min_fraction = 0.05", "random_min_fraction = 0.99")
        )
        self.check_rejected(
            text.replace(
                'confirmatory_repetition = "faithful"',
                'confirmatory_repetition = "compressed"',
            )
        )
        self.check_rejected(text.replace("[smoke]", "[smok]"))
        self.check_rejected(text.replace("depth = 16", "depth = 0"))


class SelectionTests(unittest.TestCase):
    def write_source(self, directory, records):
        path = Path(directory) / "roots.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return path

    def record(self, index, split, fen, repetition=1):
        return {
            "root_id": f"root-{index}",
            "split": split,
            "fen": fen,
            "repetition": repetition,
            "position_key": " ".join(fen.split()[:4]),
            "source": {"row": index},
        }

    def test_selection_is_stratified_disjoint_and_reproducible(self):
        records = [
            self.record(
                0, "train", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            ),
            self.record(
                1, "train", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
            ),
            self.record(
                2, "train", "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"
            ),
            self.record(
                3,
                "train",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                5,
            ),
            self.record(
                4, "val", "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_source(directory, records)
            selected, selection = select_roots(source, 3, 640)
            self.assertEqual(selection["counts"], {"train": 2, "val": 1})
            self.assertEqual({root["split"] for root in selected}, {"train", "val"})
            train_keys = {
                root["position_key"] for root in selected if root["split"] == "train"
            }
            val_keys = {
                root["position_key"] for root in selected if root["split"] == "val"
            }
            self.assertFalse(train_keys & val_keys)
            self.assertNotIn("root-2", {root["root_id"] for root in selected})
            self.assertNotIn("root-3", {root["root_id"] for root in selected})
            self.assertEqual(selection["source_sha256"], sha256(source))
            self.assertEqual(selection["source_counts"], {"train": 2, "val": 1})
            again, _ = select_roots(source, 3, 640)
            self.assertEqual(
                [root["root_id"] for root in selected],
                [root["root_id"] for root in again],
            )

    def test_requested_total_is_split_proportionally(self):
        def rank(file):
            return f"{file or ''}K{7 - file or ''}"

        records = [
            self.record(
                index,
                "train",
                f"7k/8/8/8/8/8/8/{rank(index)} w - - 0 {index + 1}",
            )
            for index in range(6)
        ]
        records += [
            self.record(
                10 + index,
                "val",
                f"7k/8/8/8/8/8/8/{rank(index)} b - - 0 {index + 1}",
            )
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_source(directory, records)
            selected, selection = select_roots(source, 2, 640)
            self.assertEqual(selection["counts"], {"train": 1, "val": 1})
            self.assertEqual(len(selected), 2)

    def test_rejects_a_terminal_root_that_survives_sampling(self):
        records = [
            self.record(0, "train", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_source(directory, records)
            with self.assertRaises(CollectionError):
                select_roots(source, 1, 640)


class CollectorIntegrationTests(unittest.TestCase):
    def test_sidecar_is_byte_identical_to_the_recorded_fixture(self):
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
            config_path = directory / "trace_v2.toml"
            config_path.write_text(
                config_text(depth=4, engine=str(Path(engine).resolve())),
                encoding="utf-8",
            )
            config = load_config(config_path)
            source = directory / "source.jsonl"
            source.write_text(json.dumps(root) + "\n", encoding="utf-8")
            roots, selection = select_roots(source, 1, 640)
            result = collect(
                roots,
                directory / "run",
                config,
                Path(patch),
                Path(control),
                Path("docs/teacher.md"),
                selection,
            )
            self.assertEqual(result["roots"], 1)
            sidecar = directory / "run" / "roots" / root["root_id"] / "sidecar.jsonl"
            self.assertEqual(sha256(sidecar), sha256(fixture / "sidecar.jsonl"))
            collection = json.loads(
                (directory / "run" / "collection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(collection["teacher"]["schema_version"], 3)
            self.assertEqual(collection["search"]["depth"], 4)
            again = collect(
                roots,
                directory / "run",
                config,
                Path(patch),
                Path(control),
                Path("docs/teacher.md"),
                selection,
            )
            self.assertEqual(again["pending"], 0)


if __name__ == "__main__":
    unittest.main()
