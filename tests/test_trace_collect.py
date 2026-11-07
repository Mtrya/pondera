import gc
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

import chess
from trace_fixtures import read_release_rows
from trace_fixtures import trace_config_text as config_text

from pondera.data.config import load_config, parse_recipe
from pondera.data.pipeline import (
    CollectionError,
    EngineStartupError,
    EngineTimeoutError,
    _check_sidecar,
    _collect_root,
    _Engine,
    _start_engine,
    produce_shards,
    run_pipeline,
    sample_root_settings,
    validate_release,
)
from pondera.data.roots.pool import read_roots, root_id_for
from pondera.data.trace import SynthesisError, load_sidecar

FIXTURES = Path(__file__).parent / "fixtures" / "trace_v2"


class RecipeTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.toml"
            path.write_text(text, encoding="utf-8")
            return load_config(path)

    def test_root_collection_restores_gc_on_success_and_producer_failure(self):
        config = self.load(config_text())
        root = json.loads((FIXTURES / "start_d4" / "root.json").read_text())
        root = sample_root_settings([root], config, config.seed)[0]
        lines = (FIXTURES / "start_d4" / "sidecar.jsonl").read_text().splitlines()
        original = gc.isenabled()
        try:
            for enabled in (True, False):
                with self.subTest(enabled=enabled):
                    (gc.enable if enabled else gc.disable)()
                    _collect_root(
                        root, config, None, None, None, {root["root_id"]: lines}
                    )
                    self.assertEqual(gc.isenabled(), enabled)
                    with self.assertRaises(SynthesisError):
                        _collect_root(
                            root,
                            config,
                            None,
                            None,
                            None,
                            {root["root_id"]: lines[:-1]},
                        )
                    self.assertEqual(gc.isenabled(), enabled)
        finally:
            (gc.enable if original else gc.disable)()

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


FAKE_ENGINE = """
import fcntl
import json
import os
import sys
import time
from pathlib import Path

manifest = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))
cases = manifest["cases"]
state_path = Path(manifest["state"])


def send(line):
    sys.stdout.write(line + "\\n")
    sys.stdout.flush()


def bump(name):
    with open(state_path.with_suffix(".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.exists()
            else {}
        )
        state[name] = state.get(name, 0) + 1
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return state[name]


launch = bump("starts")
if manifest.get("pids"):
    with open(manifest["pids"], "a") as log:
        log.write(str(os.getpid()) + "\\n")
if launch in manifest.get("bad_starts", []):
    sys.exit(1)

fen = None
trace = ""
for raw in sys.stdin:
    line = raw.strip()
    if line == "uci":
        send("id name fixture-engine")
        send("option name TraceLog type string default ")
        send("uciok")
    elif line.startswith("setoption name TraceLog value"):
        trace = line.split("value", 1)[1].strip()
    elif line.startswith("setoption"):
        optionslog = manifest.get("optionslog")
        if optionslog:
            with open(optionslog, "a") as log:
                log.write(line + "\\n")
    elif line == "isready":
        send("readyok")
    elif line.startswith("position fen "):
        fen = line[len("position fen "):]
    elif line.startswith("go "):
        with open(manifest["golog"], "a") as log:
            log.write(fen + "\\n")
        case = cases[fen]
        if fen == manifest.get("unblock_fen"):
            Path(manifest["gate"]).touch()
        if fen in manifest.get("blocked_fens", []):
            while not Path(manifest["gate"]).exists():
                time.sleep(0.01)
        if fen == manifest.get("hang_fen"):
            time.sleep(3600)
        if fen == manifest.get("fail_fen"):
            send("bestmove")
            continue
        if bump("faults") == 1 and manifest.get("fault", "die") != "none":
            fault = manifest.get("fault", "die")
            if fault == "garbage":
                Path(trace).write_text('{"malformed": true}\\n', encoding="utf-8")
                send("bestmove " + case["bestmove"])
                continue
            if fault in {"missing_fen", "bad_depth", "missing_rank"}:
                events = [json.loads(line) for line in Path(case["sidecar"]).read_text().splitlines()]
                for event in events:
                    if fault == "missing_fen" and event["type"] == "position":
                        del event["fen"]
                    if fault == "bad_depth" and event["type"] == "search_start":
                        event["depth"] = "invalid"
                    if fault == "missing_rank" and event["type"] == "round_end":
                        del event["rank"]
                Path(trace).write_text("\\n".join(json.dumps(event) for event in events) + "\\n")
                send("bestmove " + case["bestmove"])
                continue
            send("bestmove")
            sys.exit(1)
        Path(trace).write_text(
            Path(case["sidecar"]).read_text(encoding="utf-8"), encoding="utf-8"
        )
        send("bestmove " + case["bestmove"])
    elif line == "quit":
        break
"""


def fixture_engine(directory, manifest_fields=None, timeout=None, extra_cases=()):
    directory = Path(directory)
    cases = {}
    roots = []
    for name in (
        "start_d4",
        "middlegame_black_d4",
        "mate_in_1_d4",
        "claim_draw_d4",
        *extra_cases,
    ):
        fixture = FIXTURES / name
        root = json.loads((fixture / "root.json").read_text(encoding="utf-8"))
        lines = (fixture / "sidecar.jsonl").read_text(encoding="utf-8")
        cases[root["fen"]] = {
            "sidecar": str(fixture / "sidecar.jsonl"),
            "bestmove": json.loads(lines.splitlines()[-1])["bestmove"],
        }
        roots.append(root)
    script = directory / "fake_engine.py"
    golog = directory / "fake-engine-gos.jsonl"
    script.with_suffix(".json").write_text(
        json.dumps(
            {
                "cases": cases,
                "state": str(directory / "fake-engine-state.json"),
                "golog": str(golog),
                **(manifest_fields or {}),
            }
        ),
        encoding="utf-8",
    )
    script.write_text(f"#!{sys.executable}\n" + FAKE_ENGINE, encoding="utf-8")
    script.chmod(0o755)
    config_path = directory / "trace.toml"
    config_path.write_text(config_text(timeout=timeout), encoding="utf-8")
    return load_config(config_path), roots, script, golog


class EngineFailureIntegrationTests(unittest.TestCase):
    """Engine-side failures isolate their root and never escape the run."""

    def run_engine(self, directory, manifest_fields=None, timeout=None):
        directory = Path(directory)
        config, roots, script, golog = fixture_engine(
            directory, manifest_fields, timeout
        )
        result = produce_shards(
            roots,
            directory / "release",
            config,
            shard_size=2,
            workers=2,
            engine=script,
        )
        searches = Counter(golog.read_text(encoding="utf-8").splitlines())
        return result, roots, searches

    def test_a_dying_engine_is_rebuilt_and_its_root_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            result, roots, searches = self.run_engine(directory)
            self.assertEqual(result["roots"], len(roots))
            self.assertTrue(
                validate_release(Path(directory) / "release", roots, workers=1)["ok"]
            )
            # The engine died once: exactly one root was searched twice (its
            # retry). Without the worker rebuilding its engine, the dead
            # process would also fail the next roots assigned to that worker
            # within the same pass.
            self.assertEqual(sorted(searches.values()), [1] * (len(roots) - 1) + [2])

    def test_a_malformed_sidecar_is_isolated_and_retried(self):
        for fault in ("garbage", "missing_fen", "bad_depth", "missing_rank"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                result, roots, searches = self.run_engine(directory, {"fault": fault})
                self.assertEqual(result["roots"], len(roots))
                self.assertEqual(result["dropped"], 0)
                self.assertTrue(
                    validate_release(Path(directory) / "release", roots, workers=1)[
                        "ok"
                    ]
                )
                self.assertEqual(
                    sorted(searches.values()), [1] * (len(roots) - 1) + [2]
                )

    def test_later_shards_unblock_slow_roots_in_both_active_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config, roots, script, golog = fixture_engine(
                directory, timeout=5, extra_cases=("stalemate_d4",)
            )
            manifest = json.loads(script.with_suffix(".json").read_text())
            manifest.update(
                fault="none",
                blocked_fens=[roots[0]["fen"], roots[2]["fen"]],
                unblock_fen=roots[4]["fen"],
                gate=str(directory / "gate"),
            )
            script.with_suffix(".json").write_text(json.dumps(manifest))
            result = produce_shards(
                roots,
                directory / "release",
                config,
                shard_size=2,
                workers=4,
                engine=script,
            )
            self.assertEqual(result["roots"], len(roots))
            self.assertEqual(result["dropped"], 0)
            self.assertEqual(len(golog.read_text().splitlines()), len(roots))
            self.assertTrue(
                validate_release(directory / "release", roots, workers=1)["ok"]
            )

    def test_a_failed_start_aborts_without_charging_unsearched_roots(self):
        for workers in (1, 2):
            with (
                self.subTest(workers=workers),
                tempfile.TemporaryDirectory() as directory,
            ):
                directory = Path(directory)
                config, roots, script, golog = fixture_engine(
                    directory,
                    {"bad_starts": [2] if workers == 1 else list(range(1, 9))},
                )
                with self.assertRaises(EngineStartupError):
                    produce_shards(
                        roots,
                        directory / "release",
                        config,
                        shard_size=2,
                        workers=workers,
                        engine=script,
                    )
                before = Counter(
                    golog.read_text().splitlines() if golog.exists() else []
                )
                manifest = json.loads(script.with_suffix(".json").read_text())
                manifest.update(bad_starts=[], fault="none")
                script.with_suffix(".json").write_text(json.dumps(manifest))
                produce_shards(
                    roots,
                    directory / "release",
                    config,
                    shard_size=2,
                    workers=workers,
                    engine=script,
                )
                searches = Counter(golog.read_text().splitlines())
                self.assertEqual(
                    sorted(searches.values()), [1, 1, 1, 2] if workers == 1 else [1] * 4
                )
                self.assertEqual(sum(before.values()), 1 if workers == 1 else 0)

    def test_the_failure_limit_stops_hanging_engines_and_keeps_completed_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pids = directory / "pids"
            config, fixtures, script, golog = fixture_engine(
                directory, {"fault": "none", "pids": str(pids)}, timeout=60
            )
            roots = fixtures[:2] + [
                {**fixtures[2], "root_id": f"failure-{i}"} for i in range(70)
            ]
            manifest = json.loads(script.with_suffix(".json").read_text())
            manifest.update(hang_fen=roots[1]["fen"], fail_fen=roots[2]["fen"])
            script.with_suffix(".json").write_text(json.dumps(manifest))
            config_path = directory / "trace.toml"
            config_path.write_text(
                config_path.read_text().replace("workers = 8", "workers = 2")
            )
            input_path = directory / "roots.jsonl"
            input_path.write_text("".join(json.dumps(root) + "\n" for root in roots))
            roots = read_roots(input_path)
            command = [
                sys.executable,
                "-m",
                "pondera.data",
                "curate",
                "--config",
                str(config_path),
                "--input",
                str(input_path),
                "--output",
                str(directory / "release"),
                "--engine",
                str(script),
            ]
            with subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            ) as process:
                try:
                    process.communicate(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()
                    self.fail(
                        "Curation waited for a hanging search after the failure limit"
                    )
            self.assertNotEqual(process.returncode, 0)
            self.assertGreaterEqual(
                Counter(golog.read_text().splitlines())[roots[2]["fen"]], 65
            )
            self.assertEqual(
                Counter(golog.read_text().splitlines())[roots[0]["fen"]], 1
            )
            for pid in pids.read_text().splitlines():
                with self.assertRaises(ProcessLookupError):
                    os.kill(int(pid), 0)
            manifest.update(hang_fen=None, fail_fen=None)
            script.with_suffix(".json").write_text(json.dumps(manifest))
            result = run_pipeline(
                load_config(config_path),
                input_path,
                directory / "release",
                engine=script,
            )
            self.assertEqual(result["roots"], len(roots))
            self.assertEqual(
                Counter(golog.read_text().splitlines())[roots[0]["fen"]], 1
            )
            self.assertTrue(
                validate_release(directory / "release", roots, workers=1)["ok"]
            )


HANG_ENGINE = """
import sys
import time

for raw in sys.stdin:
    line = raw.strip()
    if line == "uci":
        sys.stdout.write("option name TraceLog type string default \\nuciok\\n")
        sys.stdout.flush()
    elif line == "isready":
        sys.stdout.write("readyok\\n")
        sys.stdout.flush()
    elif line.startswith("go "):
        time.sleep(3600)
    elif line == "quit":
        break
"""

SLOW_ENGINE = """
import time

time.sleep(3600)
"""


class EngineTimeoutTests(unittest.TestCase):
    """A search past the deadline raises a timeout; a slow start must not."""

    def test_a_search_past_the_deadline_raises_a_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            script = directory / "hang_engine.py"
            script.write_text(f"#!{sys.executable}\n" + HANG_ENGINE, encoding="utf-8")
            script.chmod(0o755)
            with _Engine(script, 256, timeout=1) as engine:
                with self.assertRaises(EngineTimeoutError):
                    engine.search(chess.STARTING_FEN, directory / "trace.jsonl", 1, 4)

    def test_a_slow_engine_start_is_a_plain_collection_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            script = directory / "slow_engine.py"
            script.write_text(f"#!{sys.executable}\n" + SLOW_ENGINE, encoding="utf-8")
            script.chmod(0o755)
            config_path = directory / "trace.toml"
            config_path.write_text(config_text(timeout=1), encoding="utf-8")
            config = load_config(config_path)
            with self.assertRaises(CollectionError) as caught:
                _start_engine(script, config)
            self.assertNotIsInstance(caught.exception, EngineTimeoutError)


class SyzygyOptionIntegrationTests(unittest.TestCase):
    """The engine process must receive the configured tablebase options."""

    def produce(self, directory, options):
        directory = Path(directory)
        directory.mkdir(parents=True)
        cases = {}
        roots = []
        for name in ("start_d4", "middlegame_black_d4"):
            fixture = FIXTURES / name
            root = json.loads((fixture / "root.json").read_text(encoding="utf-8"))
            lines = (fixture / "sidecar.jsonl").read_text(encoding="utf-8")
            cases[root["fen"]] = {
                "sidecar": str(fixture / "sidecar.jsonl"),
                "bestmove": json.loads(lines.splitlines()[-1])["bestmove"],
            }
            roots.append(root)
        script = directory / "fake_engine.py"
        optionslog = directory / "fake-engine-options.log"
        manifest = script.with_suffix(".json")
        manifest.write_text(
            json.dumps(
                {
                    "cases": cases,
                    "state": str(directory / "fake-engine-state.json"),
                    "golog": str(directory / "fake-engine-gos.jsonl"),
                    "optionslog": str(optionslog),
                }
            ),
            encoding="utf-8",
        )
        script.write_text(f"#!{sys.executable}\n" + FAKE_ENGINE, encoding="utf-8")
        script.chmod(0o755)
        config_path = directory / "trace.toml"
        text = config_text()
        if options is not None:
            text = text.replace("[search]\n", f"[search]\n{options}\n", 1)
        config_path.write_text(text, encoding="utf-8")
        config = load_config(config_path)
        produce_shards(
            roots,
            directory / "release",
            config,
            shard_size=2,
            workers=1,
            engine=script,
        )
        return optionslog.read_text(encoding="utf-8")

    def test_syzygy_options_reach_the_engine_only_when_configured(self):
        with tempfile.TemporaryDirectory() as directory:
            tables = Path(directory) / "tb"
            tables.mkdir()
            (tables / "KRvK.rtbw").write_bytes(b"")
            sent = self.produce(
                Path(directory) / "enabled",
                f"syzygy_path = {json.dumps(str(tables))}\nsyzygy_probe_limit = 5",
            )
            self.assertIn(f"setoption name SyzygyPath value {tables}", sent)
            self.assertIn("setoption name SyzygyProbeLimit value 5", sent)
            disabled = self.produce(Path(directory) / "disabled", None)
            self.assertNotIn("Syzygy", disabled)

    def test_an_unusable_tablebase_path_fails_before_any_search(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.produce(
                    Path(directory) / "enabled",
                    'syzygy_path = "/nonexistent/tb"\nsyzygy_probe_limit = 5',
                )


class DirectPipelineIntegrationTests(unittest.TestCase):
    def engine(self):
        engine = os.environ.get("PONDERA_TRACE_ENGINE")
        if not engine or not Path(engine).is_file():
            self.skipTest("set PONDERA_TRACE_ENGINE to the instrumented engine")
        return Path(engine).resolve()

    def test_new_game_resets_match_fresh_recorded_searches(self):
        engine = self.engine()
        cases = (
            ("start_d4", 1),
            ("middlegame_black_w3", 3),
            ("mate_in_1_w3", 3),
            ("start_d4", 1),
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            _Engine(engine, 256) as session,
        ):
            for index, (case, width) in enumerate(cases):
                fixture = FIXTURES / case
                root = json.loads((fixture / "root.json").read_text())
                path = Path(directory) / f"{index}.jsonl"
                session.search(root["fen"], path, width, 4)
                actual = [json.loads(line) for line in path.read_text().splitlines()]
                expected = [
                    json.loads(line)
                    for line in (fixture / "sidecar.jsonl").read_text().splitlines()
                ]
                self.assertEqual(actual, expected, case)

    def test_engine_produces_a_release_without_a_sidecar_directory(self):
        engine = self.engine()
        fixture = FIXTURES / "start_d4"
        root = json.loads((fixture / "root.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config_path = directory / "trace.toml"
            config_path.write_text(
                config_text().replace("workers = 8", "workers = 1"),
                encoding="utf-8",
            )
            (directory / "roots.jsonl").write_text(
                json.dumps(root) + "\n", encoding="utf-8"
            )
            before = (directory / "roots.jsonl").read_bytes()
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pondera.data",
                    "curate",
                    "--config",
                    str(config_path),
                    "--input",
                    str(directory),
                    "--output",
                    str(directory / "release"),
                    "--engine",
                    str(engine),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(completed.stdout)
            self.assertEqual((directory / "roots.jsonl").read_bytes(), before)
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
            path.write_text(config_text(joint=joint))
            config = load_config(path)
            sampled = sample_root_settings(roots, config, config.seed)
            self.assertEqual({r["depth"] for r in sampled}, {3, 4})
            self.assertIn(1, {r["teacher_width"] for r in sampled})
            self.assertIn(2, {r["teacher_width"] for r in sampled})
            produce_shards(
                roots,
                directory / "serial",
                config,
                engine=engine,
                workers=1,
                shard_size=2,
            )
            produce_shards(
                list(reversed(roots)),
                directory / "parallel",
                config,
                engine=engine,
                workers=2,
                shard_size=3,
            )
            partial = produce_shards(
                roots,
                directory / "resumed",
                config,
                engine=engine,
                workers=2,
                shard_size=2,
                end_shard=1,
            )
            self.assertEqual(partial["pending"], 4)
            # Recover a real published shard whose registration was interrupted.
            (directory / "resumed" / "release.json").unlink()
            produce_shards(
                roots,
                directory / "resumed",
                config,
                engine=engine,
                workers=2,
                shard_size=2,
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
            produce_shards(
                roots[-1:], directory / "full-one", full, engine=engine, workers=1
            )
            self.assertTrue(validate_release(directory / "full-one", workers=1)["ok"])
            self.assertEqual(
                next(iter(read_release_rows(directory / "full-one").values()))["row"][
                    "teacher_width"
                ],
                1,
            )


if __name__ == "__main__":
    unittest.main()
