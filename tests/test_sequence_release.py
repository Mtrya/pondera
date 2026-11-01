import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from trace_fixtures import tie_sidecar_lines

from pondera.chess.symmetry import reflect_uci
from pondera.chess.vocabulary import MOVE, UCI_MOVE_TO_IDX
from pondera.data.positions import SequencePositions
from pondera.data.sequence_release import load_collection, produce_shards
from pondera.data.sequence_validation import (
    SEQUENCE_ARRAYS,
    load_sequence_arrays,
    sequence_data_identity,
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
ENGINE_SHA256 = "52498f1a544a7c238c9c6db0614aa5821877b83d157db167d931ec56f4525761"
WIDE_ENGINE_SHA256 = "1420c314729fa3d40d6e2fda2834d44f422902d9b0d37c5d0fe1bf5d5f1f845b"
INSTRUMENTATION_SHA256 = (
    "a8ff707418c820f6f5ad411a2744141286c5d86e4b67553f2373a8c6f4128a67"
)
WIDE_INSTRUMENTATION_SHA256 = (
    "4edea22f905407cbc749e3bc8c3784ecb5de781ec1ed54e37c6dc23b42af3b53"
)
CONTROL_SHA256 = "3116251d2cdf84f710b7fe27a037d0b0a390f1f379fb1f32aeace2f41f095b34"


def config_text(width=1):
    text = (
        Path("configs/trace_v2.toml")
        .read_text(encoding="utf-8")
        .replace("depth = 16", "depth = 4")
    )
    if width != 1:
        text = text.replace("teacher_width = 1", f"teacher_width = {width}")
    return text


def build_run(directory, cases=CASES, width=1, sidecar_mutator=None):
    directory = Path(directory)
    config_path = directory / "trace_v2.toml"
    config_path.write_text(config_text(width), encoding="utf-8")
    config = load_config(config_path)
    run = directory / "run"
    (run / "roots").mkdir(parents=True)
    roots = []
    mapping = {}
    for case in cases:
        fixture = FIXTURES / case
        root = json.loads((fixture / "root.json").read_text(encoding="utf-8"))
        roots.append(root)
        target = run / "roots" / root["root_id"]
        target.mkdir()
        if sidecar_mutator is None:
            shutil.copyfile(fixture / "sidecar.jsonl", target / "sidecar.jsonl")
        else:
            lines = (fixture / "sidecar.jsonl").read_text(encoding="utf-8").splitlines()
            target.joinpath("sidecar.jsonl").write_text(
                "\n".join(sidecar_mutator(case, lines)) + "\n", encoding="utf-8"
            )
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
    identity = {
        "teacher": {
            "schema_version": 3 if width == 1 else 4,
            "contract": "docs/teacher.md",
            "contract_sha256": sha256("docs/teacher.md"),
            "value_mate": session["value_mate"],
            "max_ply": session["max_ply"],
        },
        "engine": {
            "name": session["engine"],
            "sha256": ENGINE_SHA256 if width == 1 else WIDE_ENGINE_SHA256,
            "instrumentation": "instrumentation.patch",
            "instrumentation_sha256": INSTRUMENTATION_SHA256
            if width == 1
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
            "seed": 640,
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
    collection_path = run / "collection.json"
    write_manifest(
        collection_path,
        {"kind": "trace-v2-collection", "format_version": 1, **identity},
    )
    return config, roots, collection_path, run


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
