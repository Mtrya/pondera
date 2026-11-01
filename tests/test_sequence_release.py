import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
from trace_fixtures import tie_sidecar_lines

from pondera.chess.symmetry import reflect_uci
from pondera.chess.vocabulary import MOVE, UCI_MOVE_TO_IDX
from pondera.data.positions import SequencePositions, sequence_data_identity
from pondera.data.sequence_format import SEQUENCE_ARRAYS
from pondera.data.sequence_release import load_collection, produce_shards
from pondera.data.sequence_validation import (
    load_sequence_arrays,
    validate_release,
    validate_release_identity,
    verify_shard_files,
)
from pondera.data.storage import content_sha256, load_manifest, sha256, write_manifest
from pondera.data.trace import replay
from pondera.data.trace_config import load_config
from pondera.data.trace_synthesis import PRESENTATIONS

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
widths = {1 = 0.5, 3 = 0.5}
depths = {4 = 1.0}
presentations = {zero = 0.2, full = 0.5, random = 0.3}
"""
ENGINE_SHA256 = "52498f1a544a7c238c9c6db0614aa5821877b83d157db167d931ec56f4525761"
WIDE_ENGINE_SHA256 = "1420c314729fa3d40d6e2fda2834d44f422902d9b0d37c5d0fe1bf5d5f1f845b"
INSTRUMENTATION_SHA256 = (
    "a8ff707418c820f6f5ad411a2744141286c5d86e4b67553f2373a8c6f4128a67"
)
WIDE_INSTRUMENTATION_SHA256 = (
    "4edea22f905407cbc749e3bc8c3784ecb5de781ec1ed54e37c6dc23b42af3b53"
)
CONTROL_SHA256 = "3116251d2cdf84f710b7fe27a037d0b0a390f1f379fb1f32aeace2f41f095b34"


def config_text(width=1, recipe_text=None):
    text = (
        Path("configs/trace_v2.toml")
        .read_text(encoding="utf-8")
        .replace("depth = 16", "depth = 4")
    )
    if width != 1:
        text = text.replace("teacher_width = 1", f"teacher_width = {width}")
    if recipe_text is not None:
        text = text.rstrip() + "\n\n" + recipe_text.strip() + "\n"
    return text


def build_run(
    directory, cases=CASES, width=1, sidecar_mutator=None, recipe_text=None, seed=640
):
    directory = Path(directory)
    config_path = directory / "trace_v2.toml"
    config_path.write_text(config_text(width, recipe_text), encoding="utf-8")
    config = load_config(config_path)
    run = directory / "run"
    (run / "roots").mkdir(parents=True)
    roots = []
    mapping = {}
    for case in cases:
        fixture = FIXTURES / case
        root = json.loads((fixture / "root.json").read_text(encoding="utf-8"))
        target = run / "roots" / root["root_id"]
        target.mkdir()
        if sidecar_mutator is None:
            shutil.copyfile(fixture / "sidecar.jsonl", target / "sidecar.jsonl")
        else:
            lines = (fixture / "sidecar.jsonl").read_text(encoding="utf-8").splitlines()
            target.joinpath("sidecar.jsonl").write_text(
                "\n".join(sidecar_mutator(case, lines)) + "\n", encoding="utf-8"
            )
        if recipe_text is not None:
            session = json.loads(
                (target / "sidecar.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            root["teacher_width"] = session["options"]["multipv"]
            root["depth"] = config.search.depth
        roots.append(root)
        mapping[root["root_id"]] = sha256(target / "sidecar.jsonl")
    roots_path = run / "roots.jsonl"
    roots_path.write_text(
        "".join(json.dumps(root, separators=(",", ":")) + "\n" for root in roots),
        encoding="utf-8",
    )
    write_manifest(run / "sidecars.json", mapping)
    session = json.loads(
        (FIXTURES / cases[0] / "sidecar.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    widths = [root.get("teacher_width", config.search.teacher_width) for root in roots]
    schema = 4 if max(widths) > 1 else 3
    identity = {
        "teacher": {
            "schema_version": schema,
            "contract": "docs/teacher.md",
            "contract_sha256": sha256("docs/teacher.md"),
            "value_mate": session["value_mate"],
            "max_ply": session["max_ply"],
        },
        "engine": {
            "name": session["engine"],
            "sha256": ENGINE_SHA256 if schema == 3 else WIDE_ENGINE_SHA256,
            "instrumentation": "instrumentation.patch",
            "instrumentation_sha256": INSTRUMENTATION_SHA256
            if schema == 3
            else WIDE_INSTRUMENTATION_SHA256,
            "control": {"path": "control", "sha256": CONTROL_SHA256},
        },
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "search": config.search_dict(),
        "board": config.board_dict(),
        "presentation": config.presentation_dict(),
        "synthesis": config.synthesis_dict(),
        "root_selection": {
            "source": str(roots_path),
            "source_sha256": sha256(roots_path),
            "algorithm": "fixture",
            "seed": seed,
            "requested": len(roots),
            "counts": {"train": len(roots), "val": 0},
            "source_counts": {"train": len(roots), "val": 0},
        },
        "roots": {
            "path": "roots.jsonl",
            "rows": len(roots),
            "sha256": content_sha256(roots),
        },
        "sidecars": {"rows": len(mapping), "sha256": content_sha256(mapping)},
    }
    if config.recipe is not None:
        identity["recipe"] = config.recipe.identity()
    collection_path = run / "collection.json"
    write_manifest(
        collection_path,
        {"kind": "trace-v2-collection", "format_version": 1, **identity},
    )
    return config, roots, collection_path, run


def read_release_rows(directory):
    directory = Path(directory)
    release = load_manifest(directory / "release.json")
    output = {}
    for shard in release["shards"]:
        _, rows, arrays = verify_shard_files(directory, shard)
        for index, row in enumerate(rows):
            length = int(arrays["trace_len"][index])
            count = int(arrays["state_count"][index])
            output[(row["root_id"], row["presentation"])] = {
                "row": row,
                "trace_tokens": tuple(
                    int(value) for value in arrays["trace_tokens"][index, :length]
                ),
                "trace_roles": tuple(
                    int(value) for value in arrays["trace_roles"][index, :length]
                ),
                "state": {
                    name: tuple(int(value) for value in arrays[name][index, :count])
                    for name in (
                        "state_best",
                        "state_value",
                        "state_basis",
                        "state_no_tool",
                        "state_tool",
                    )
                },
            }
    return output


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (
            self.config,
            self.roots,
            self.collection_path,
            self.run,
        ) = build_run(self.root)
        _collection, selected, identity, base = load_collection(self.collection_path)
        self.assertEqual(selected, self.roots)
        self.result = produce_shards(
            self.roots,
            base,
            base / "release",
            self.config,
            identity,
            contract_path=Path("docs/teacher.md"),
            shard_size=2,
            workers=1,
        )
        self.release = base / "release"

    def test_release_layout_identity_and_validation(self):
        self.assertEqual(self.result["roots"], len(CASES))
        # Five roots, shard_size two: 2 + 2 + 1 roots, 8 + 8 + 4 rows.
        self.assertEqual(len(self.result["shards"]), 3)
        self.assertEqual([shard["rows"] for shard in self.result["shards"]], [8, 8, 4])
        release = load_manifest(self.release / "release.json")
        self.assertEqual(release["format_version"], 5)
        self.assertEqual(release["grammar_version"], 2)
        self.assertEqual(release["teacher"]["schema_version"], 3)
        self.assertEqual(
            release["teacher"]["contract_sha256"], sha256("docs/teacher.md")
        )
        self.assertEqual(release["search"]["depth"], 4)
        self.assertEqual(release["search"]["teacher_width"], 1)
        self.assertEqual(release["engine"]["sha256"], ENGINE_SHA256)
        self.assertEqual(release["presentation"], self.config.presentation_dict())
        self.assertEqual(release["roots"]["rows"], len(CASES))
        self.assertEqual(release["sidecars"]["rows"], len(CASES))
        self.assertEqual(release["metadata"], validate_release_identity(release))
        validation = validate_release(self.release, self.roots, workers=1)
        self.assertTrue(validation["ok"])
        self.assertEqual(validation["roots"], len(CASES))
        self.assertEqual(validation["rows"], len(CASES) * len(PRESENTATIONS))
        self.assertEqual(validation["counts"]["train"], len(CASES) * len(PRESENTATIONS))
        self.assertTrue((self.release / "validation.json").exists())

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
                "kind",
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
            self.assertIn("champions.jsonl", manifest["files"])
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
                tokens = arrays["trace_tokens"][index, :length]
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
                self.assertEqual(row["thinking_tokens"], final.start)
                self.assertEqual(len(champions[key]["rounds"]), row["depth"])
                thought = [
                    step
                    for step in program.steps
                    if step.operation
                    not in {"final", "final_evaluate", "final_pv", "end"}
                ]
                self.assertEqual(int(arrays["state_count"][index]), len(thought))
        self.assertEqual(set(suffixes), {root["root_id"] for root in self.roots})
        positions = SequencePositions(self.release, "train")
        self.assertEqual(len(positions), len(CASES) * len(PRESENTATIONS))
        row = positions[0]
        self.assertEqual(len(row), 4)
        self.assertGreater(int(row[2]), 0)

    def test_resume_and_shard_recovery(self):
        _, _, identity, _ = load_collection(self.collection_path)
        again = produce_shards(
            self.roots,
            self.run,
            self.release,
            self.config,
            identity,
            contract_path=Path("docs/teacher.md"),
            shard_size=2,
            workers=1,
        )
        self.assertEqual(again["roots"], len(CASES))
        self.assertEqual(len(again["shards"]), 3)
        (self.release / "release.json").unlink()
        recovered = produce_shards(
            self.roots,
            self.run,
            self.release,
            self.config,
            identity,
            contract_path=Path("docs/teacher.md"),
            shard_size=2,
            workers=1,
        )
        self.assertEqual(recovered["roots"], len(CASES))
        self.assertTrue(validate_release(self.release, self.roots, workers=1)["ok"])

    def test_sequence_data_identity_covers_the_training_inputs(self):
        identity = sequence_data_identity(self.release)
        self.assertEqual(len(identity), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in identity))

    def test_validation_detects_a_tampered_shard(self):
        release = load_manifest(self.release / "release.json")
        shard = release["shards"][0]
        path = self.release / shard["path"] / "trace_tokens.npy"
        with path.open("r+b") as handle:
            handle.seek(path.stat().st_size // 2)
            byte = handle.read(1)
            handle.seek(-1, 1)
            handle.write(bytes([byte[0] ^ 0xFF]))
        with self.assertRaises(ValueError):
            validate_release(self.release, self.roots, workers=1)

    def test_identity_rejects_inconsistent_metadata(self):
        release = load_manifest(self.release / "release.json")
        cases = []
        wrong = copy.deepcopy(release)
        wrong["metadata"]["search"]["teacher_width"] = 2
        cases.append(wrong)
        wrong = copy.deepcopy(release)
        wrong["metadata"]["presentation"]["zero"] = 0.2
        cases.append(wrong)
        wrong = copy.deepcopy(release)
        wrong["metadata"]["teacher"]["schema_version"] = 2
        cases.append(wrong)
        wrong = copy.deepcopy(release)
        wrong["metadata"]["engine"]["sha256"] = "not-a-checksum"
        cases.append(wrong)
        wrong = copy.deepcopy(release)
        wrong["metadata"]["roots"]["sha256"] = "0" * 64
        cases.append(wrong)
        for wrong in cases:
            with self.subTest(wrong=wrong["metadata"]):
                with self.assertRaises(ValueError):
                    validate_release_identity(wrong)

    def test_claim_root_is_materialized_without_a_variation(self):
        release = load_manifest(self.release / "release.json")
        claim_roots = set()
        expected = self.expected_shard_metadata(release)
        for shard in release["shards"]:
            _, rows, _ = verify_shard_files(self.release, shard, expected)
            for row in rows:
                if row["claim_draw"]:
                    claim_roots.add(row["root_id"])
                    self.assertEqual(row["final_move"], "<claim_draw>")
                    self.assertEqual(row["mate_distance"], None)
        claim = json.loads((FIXTURES / "claim_draw_d4" / "root.json").read_text())
        self.assertEqual(claim_roots, {claim["root_id"]})


class WideReleaseTests(unittest.TestCase):
    def test_wide_teacher_release_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            config, roots, collection_path, run = build_run(
                directory, cases=W3_CASES, width=3
            )
            _collection, selected, identity, base = load_collection(collection_path)
            self.assertEqual(selected, roots)
            produce_shards(
                roots,
                base,
                base / "release",
                config,
                identity,
                contract_path=Path("docs/teacher.md"),
                shard_size=1,
                workers=1,
            )
            release = load_manifest(base / "release" / "release.json")
            self.assertEqual(release["teacher"]["schema_version"], 4)
            self.assertEqual(release["search"]["teacher_width"], 3)
            self.assertEqual(release["engine"]["sha256"], WIDE_ENGINE_SHA256)
            validation = validate_release(base / "release", roots, workers=1)
            self.assertTrue(validation["ok"])
            self.assertEqual(validation["rows"], len(W3_CASES) * len(PRESENTATIONS))

    def test_mixed_width_recipe_release_and_validation(self):
        cases = ("start_d4", "middlegame_black_w3", "two_moves_w3")
        # Seed 6 samples the widths these fixtures were recorded at (1, 3, 3)
        # and the presentations frozen below.
        expected_presentations = {
            "bc6ae1d761cf700dfb66ef956ac17bef4a90e0f4846624e57d22ba65f92426c7": "random",
            "23fd2fbf6d032131267d358c35ba053a17fecb87facfabf0b3331e01e0bb8c55": "full",
            "6c9691cab19e3988acbe741a28cf1739fe971be60fdba32fe431f2e43c89a59a": "zero",
        }
        with tempfile.TemporaryDirectory() as directory:
            config, roots, collection_path, run = build_run(
                directory,
                cases=cases,
                width=1,
                recipe_text=MIXED_RECIPE,
                seed=6,
            )
            self.assertEqual(
                {root["root_id"]: root["teacher_width"] for root in roots},
                {
                    "bc6ae1d761cf700dfb66ef956ac17bef4a90e0f4846624e57d22ba65f92426c7": 1,
                    "23fd2fbf6d032131267d358c35ba053a17fecb87facfabf0b3331e01e0bb8c55": 3,
                    "6c9691cab19e3988acbe741a28cf1739fe971be60fdba32fe431f2e43c89a59a": 3,
                },
            )
            _collection, selected, identity, base = load_collection(collection_path)
            self.assertEqual(selected, roots)
            produce_shards(
                roots,
                base,
                base / "release",
                config,
                identity,
                contract_path=Path("docs/teacher.md"),
                shard_size=2,
                workers=1,
            )
            release = load_manifest(base / "release" / "release.json")
            self.assertEqual(release["teacher"]["schema_version"], 4)
            self.assertEqual(release["search"]["teacher_width"], 1)
            self.assertEqual(release["recipe"], config.recipe.identity())
            validation = validate_release(base / "release", roots, workers=1)
            self.assertTrue(validation["ok"])
            self.assertEqual(validation["rows"], len(cases))
            records = read_release_rows(base / "release")
            self.assertEqual(
                {
                    row["root_id"]: row["presentation"]
                    for row in (record["row"] for record in records.values())
                },
                expected_presentations,
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

    def test_recipe_width_one_under_a_wider_fallback_uses_schema_three(self):
        recipe = """
[recipe]
widths = {1 = 1.0}
depths = {4 = 1.0}
"""
        with tempfile.TemporaryDirectory() as directory:
            config, roots, collection_path, run = build_run(
                directory, cases=("start_d4",), width=3, recipe_text=recipe
            )
            self.assertEqual(config.search.teacher_width, 3)
            self.assertEqual({root["teacher_width"] for root in roots}, {1})
            _collection, selected, identity, base = load_collection(collection_path)
            self.assertEqual(selected, roots)
            produce_shards(
                roots,
                base,
                base / "release",
                config,
                identity,
                contract_path=Path("docs/teacher.md"),
                shard_size=1,
                workers=1,
            )
            release = load_manifest(base / "release" / "release.json")
            self.assertEqual(release["search"]["teacher_width"], 3)
            self.assertEqual(release["teacher"]["schema_version"], 3)
            validation = validate_release(base / "release", roots, workers=1)
            self.assertTrue(validation["ok"])

    def test_validation_rejects_recipe_keys_that_cannot_be_sampled(self):
        # A zero probability makes the recorded setting unreachable: the
        # recipe always draws width 3 (depth 5), never the recorded 1 (4).
        cases = (
            ("widths = {1 = 0.0, 3 = 1.0}", None),
            (None, "depths = {4 = 0.0, 5 = 1.0}"),
        )
        for widths, depths in cases:
            with self.subTest(widths=widths, depths=depths):
                recipe = "[recipe]\n"
                recipe += (
                    f"{widths}\n" if widths is not None else "widths = {1 = 1.0}\n"
                )
                recipe += (
                    f"{depths}\n" if depths is not None else "depths = {4 = 1.0}\n"
                )
                with tempfile.TemporaryDirectory() as directory:
                    config, roots, collection_path, run = build_run(
                        directory,
                        cases=("start_d4",),
                        recipe_text=recipe,
                    )
                    _collection, selected, identity, base = load_collection(
                        collection_path
                    )
                    self.assertEqual(selected, roots)
                    produce_shards(
                        roots,
                        base,
                        base / "release",
                        config,
                        identity,
                        contract_path=Path("docs/teacher.md"),
                        shard_size=1,
                        workers=1,
                    )
                    with self.assertRaises(ValueError):
                        validate_release(base / "release", workers=1)

    def test_presentation_sampling_selects_existing_rows(self):
        cases = ("start_d4", "middlegame_black_w3", "two_moves_w3")
        with tempfile.TemporaryDirectory() as plain_directory:
            with tempfile.TemporaryDirectory() as recipe_directory:
                runs = []
                for text, directory in (
                    (None, plain_directory),
                    (MIXED_RECIPE, recipe_directory),
                ):
                    config, roots, collection_path, run = build_run(
                        directory, cases=cases, recipe_text=text
                    )
                    _, _, identity, base = load_collection(collection_path)
                    produce_shards(
                        roots,
                        base,
                        base / "release",
                        config,
                        identity,
                        contract_path=Path("docs/teacher.md"),
                        shard_size=2,
                        workers=1,
                    )
                    runs.append(base / "release")
                plain = read_release_rows(runs[0])
                sampled = read_release_rows(runs[1])
                self.assertEqual(len(sampled), len(cases))
                for key, record in sampled.items():
                    self.assertEqual(record, plain[key])

    def test_tied_wide_round_releases_with_first_come_state(self):
        with tempfile.TemporaryDirectory() as directory:
            config, roots, collection_path, run = build_run(
                directory,
                cases=("middlegame_black_w3",),
                width=3,
                sidecar_mutator=tie_sidecar_lines,
            )
            _collection, selected, identity, base = load_collection(collection_path)
            self.assertEqual(selected, roots)
            produce_shards(
                roots,
                base,
                base / "release",
                config,
                identity,
                contract_path=Path("docs/teacher.md"),
                shard_size=1,
                workers=1,
            )
            validation = validate_release(base / "release", roots, workers=1)
            self.assertTrue(validation["ok"])
            shard = base / "release" / "shard-000000"
            arrays = load_sequence_arrays(shard, SEQUENCE_ARRAYS)
            self.assertEqual(arrays["trace_tokens"].dtype, np.int16)
            self.assertEqual(arrays["board_tokens"].dtype, np.int16)
            self.assertEqual(arrays["trace_roles"].dtype, np.int8)
            self.assertEqual(arrays["trace_len"].dtype, np.int32)
            self.assertEqual(arrays["state_count"].dtype, np.int32)
            for name in (
                "state_no_tool",
                "state_tool",
                "state_best",
                "state_basis",
                "state_value",
            ):
                self.assertEqual(arrays[name].dtype, np.int32)
            rows = [
                json.loads(line)
                for line in (shard / "roots.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            full = next(
                index for index, row in enumerate(rows) if row["presentation"] == "full"
            )
            count = int(arrays["state_count"][full])
            basis = arrays["state_basis"][full, :count]
            best = arrays["state_best"][full, :count]
            last_round_one = max(
                index for index in range(count) if int(basis[index]) == 1
            )
            kept = MOVE + UCI_MOVE_TO_IDX[reflect_uci("f6e4")]
            self.assertEqual(int(best[last_round_one]), kept)


if __name__ == "__main__":
    unittest.main()
