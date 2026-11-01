import dataclasses
import json
import unittest
from pathlib import Path

from pondera.chess.codec import mate_token, value_bin
from pondera.chess.tracker import PathTracker
from pondera.chess.vocabulary import DEPTH, END, FINAL, VALUE
from pondera.data.trace import DRAW_CLAIM_BOUND, replay
from pondera.data.trace_config import load_config
from pondera.data.trace_synthesis import (
    PRESENTATIONS,
    SynthesisError,
    load_sidecar,
    mate_distance_moves,
    parse_sidecar,
    profile_roots,
    synthesize_root,
)

FIXTURES = Path(__file__).parent / "fixtures" / "trace_v2"
CASES = {
    "start_d4": 4,
    "middlegame_black_d4": 4,
    "mate_in_1_d4": 4,
    "claim_draw_d4": 4,
    "capture_d4": 4,
    "rule_draw_d4": 4,
    "stalemate_d4": 4,
}
W3_CASES = {
    "middlegame_black_w3": 4,
    "mate_in_1_w3": 4,
}


def load_case(name, teacher_width=1):
    directory = FIXTURES / name
    sidecar = load_sidecar(directory / "sidecar.jsonl", teacher_width=teacher_width)
    root = json.loads((directory / "root.json").read_text(encoding="utf-8"))
    return sidecar, root


def synthesize_case(name, seed=640, teacher_width=1):
    sidecar, root = load_case(name, teacher_width)
    config = load_config()
    depth = CASES.get(name, W3_CASES.get(name))
    config = dataclasses.replace(
        config,
        search=dataclasses.replace(
            config.search, depth=depth, teacher_width=teacher_width
        ),
    )
    return sidecar, root, synthesize_root(sidecar, root, config, seed=seed)


def accepted_events(sidecar):
    attempts = {(event["round"], event["attempt"]) for event in sidecar.round_ends}
    return {
        (event["round"], event["attempt"], tuple(event["path"])): event
        for event in sidecar.candidates
        if (event["round"], event["attempt"]) in attempts
    }


def round_of(steps, index):
    round_number = 0
    for step in steps[: index + 1]:
        if step.operation == "depth":
            round_number += 1
    return round_number


class MateConversionTests(unittest.TestCase):
    """Frozen format_score values, derived by hand from the UCI rule."""

    def test_win_rounds_up_and_loss_truncates_toward_zero(self):
        cases = {
            32000: 0,
            31999: 1,
            31998: 1,
            31997: 2,
            31996: 2,
            -32000: 0,
            -31999: 0,
            -31998: -1,
            -31997: -1,
            -31996: -2,
        }
        for raw, moves in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(mate_distance_moves(raw, 32000, 246), moves)


class SynthesisTests(unittest.TestCase):
    def test_four_presentations_replay_and_share_the_suffix(self):
        for name in CASES:
            with self.subTest(case=name):
                sidecar, root, synthesized = synthesize_case(name)
                self.assertEqual(set(synthesized.rows), set(PRESENTATIONS))
                suffixes = set()
                for presentation, row in synthesized.rows.items():
                    program = replay(
                        row.tokens,
                        root["fen"],
                        root["repetition"],
                        teacher_width=1,
                    )
                    self.assertEqual(tuple(program.tokens), row.tokens)
                    self.assertEqual(tuple(program.roles), row.roles)
                    final = next(
                        step for step in program.steps if step.operation == "final"
                    )
                    self.assertEqual(row.thinking_tokens, final.start)
                    self.assertEqual(
                        row.rounds,
                        sum(1 for token in row.tokens[: final.start] if token == DEPTH),
                    )
                    suffixes.add(row.tokens[final.start :])
                    self.assertEqual(
                        row.final_move, synthesized.rows["full"].final_move
                    )
                self.assertEqual(len(suffixes), 1)
                full = synthesized.rows["full"]
                self.assertEqual(
                    synthesized.rows["zero"].tokens, full.tokens[full.thinking_tokens :]
                )
                self.assertEqual(synthesized.rows["zero"].thinking_tokens, 0)
                self.assertEqual(synthesized.rows["zero"].boundaries, ())
                self.assertEqual(
                    synthesized.rows["last_root"].tokens,
                    full.tokens[: full.boundaries[-2].tool_end]
                    + full.tokens[full.thinking_tokens :],
                )
                random = synthesized.rows["random"]
                self.assertLessEqual(random.thinking_tokens, full.thinking_tokens)
                self.assertEqual(
                    random.tokens[: random.thinking_tokens],
                    full.tokens[: random.thinking_tokens],
                )

    def test_every_refutation_maps_to_a_null_window_fail_low(self):
        for name in CASES:
            with self.subTest(case=name):
                sidecar, root, synthesized = synthesize_case(name)
                program = replay(
                    synthesized.rows["full"].tokens,
                    root["fen"],
                    root["repetition"],
                    teacher_width=1,
                )
                events = accepted_events(sidecar)
                for index, step in enumerate(program.steps):
                    if step.operation != "refuted":
                        continue
                    round_number = round_of(program.steps, index)
                    key = (
                        round_number,
                        sidecar.round_ends[round_number - 1]["attempt"],
                        step.path,
                    )
                    event = events.get(key)
                    self.assertIsNotNone(event, key)
                    self.assertIn(event["call"], {"probe", "rescore"})
                    self.assertEqual(event["bound"], "lower")
                    window = event["window"]
                    self.assertEqual(window["beta"] - window["alpha"], 1)

    def test_exact_evals_map_to_sidecar_values(self):
        for name in CASES:
            with self.subTest(case=name):
                sidecar, root, synthesized = synthesize_case(name)
                program = replay(
                    synthesized.rows["full"].tokens,
                    root["fen"],
                    root["repetition"],
                    teacher_width=1,
                )
                events = accepted_events(sidecar)
                previous_round = 0
                for index, step in enumerate(program.steps):
                    if step.operation == "depth":
                        previous_round += 1
                    if step.operation != "evaluate":
                        continue
                    token = int(program.tokens[step.start])
                    if not step.path:
                        top = sidecar.round_ends[previous_round - 1]["rank"][0]
                        if top.get("wdl") is None:
                            expected = mate_token(
                                abs(mate_distance_moves(top["raw"], 32000, 246)),
                                top["raw"] > 0,
                            )
                        else:
                            expected = VALUE + value_bin(
                                (top["wdl"][0] - top["wdl"][2]) / 1000
                            )
                        self.assertEqual(token, expected)
                        continue
                    key = (
                        previous_round,
                        sidecar.round_ends[previous_round - 1]["attempt"],
                        step.path,
                    )
                    event = events.get(key)
                    self.assertIsNotNone(event, key)
                    child = PathTracker(root["fen"], root["repetition"])
                    for move in step.path:
                        child.push(move)
                    terminal = child.search_terminal()
                    if terminal == "mate":
                        self.assertEqual(token, mate_token(0, True))
                        continue
                    if terminal == "draw":
                        self.assertEqual(token, VALUE + value_bin(0.0))
                        continue
                    self.assertEqual(event["bound"], "exact")
                    self.assertIn(event["call"], {"first", "promote"})
                    if event.get("wdl") is None:
                        expected = mate_token(
                            abs(
                                mate_distance_moves(
                                    event["raw"]
                                    if event["ply"] % 2 == 0
                                    else -event["raw"],
                                    32000,
                                    246,
                                )
                            ),
                            (event["raw"] if event["ply"] % 2 == 0 else -event["raw"])
                            > 0,
                        )
                    else:
                        scalar = (event["wdl"][0] - event["wdl"][2]) / 1000
                        if event["ply"] % 2:
                            scalar = -scalar
                        expected = VALUE + value_bin(scalar)
                    self.assertEqual(token, expected)

    def test_champions_and_final_suffix_follow_the_accepted_rounds(self):
        for name in CASES:
            with self.subTest(case=name):
                sidecar, root, synthesized = synthesize_case(name)
                self.assertEqual(
                    [audit.move for audit in synthesized.round_audits],
                    [event["rank"][0]["move"] for event in sidecar.round_ends],
                )
                full = synthesized.rows["full"]
                deepest = sidecar.round_ends[-1]["rank"][0]
                if full.claim_draw:
                    self.assertEqual(full.final_move, "<claim_draw>")
                    self.assertLessEqual(deepest["raw"], DRAW_CLAIM_BOUND)
                else:
                    self.assertEqual(full.final_move, deepest["move"])
                final_start = full.thinking_tokens
                self.assertEqual(full.tokens[final_start], FINAL)
                self.assertEqual(full.tokens[-1], END)
                if deepest.get("wdl") is None:
                    self.assertEqual(
                        full.final_value_bin, 19 if deepest["raw"] > 0 else 0
                    )
                    self.assertEqual(
                        full.mate_distance,
                        mate_distance_moves(deepest["raw"], 32000, 246),
                    )
                else:
                    self.assertEqual(
                        full.final_value_bin,
                        value_bin((deepest["wdl"][0] - deepest["wdl"][2]) / 1000),
                    )
                    self.assertIsNone(full.mate_distance)

    def test_claimable_root_with_a_nonpositive_value_claims_a_draw(self):
        sidecar, root, synthesized = synthesize_case("claim_draw_d4")
        self.assertEqual(root["repetition"], 3)
        self.assertEqual(sidecar.round_ends[-1]["rank"][0]["raw"], -18)
        full = synthesized.rows["full"]
        self.assertTrue(full.claim_draw)
        self.assertEqual(full.final_move, "<claim_draw>")
        final_start = full.thinking_tokens
        program = replay(full.tokens, root["fen"], root["repetition"], teacher_width=1)
        final = next(step for step in program.steps if step.operation == "final")
        self.assertEqual(final.uci, "<claim_draw>")
        self.assertEqual(
            [step.operation for step in program.steps if step.start >= final_start],
            ["final", "final_evaluate", "end"],
        )
        self.assertEqual(
            full.final_value_bin,
            value_bin(
                (
                    sidecar.round_ends[-1]["rank"][0]["wdl"][0]
                    - sidecar.round_ends[-1]["rank"][0]["wdl"][2]
                )
                / 1000
            ),
        )
        self.assertIsNone(full.mate_distance)

    def test_claim_gate_treats_the_draw_score_jitter_as_a_draw(self):
        sidecar, root, unmodified = synthesize_case("rule_draw_d4")
        self.assertEqual(sidecar.round_ends[-1]["rank"][0]["raw"], 1)
        # As recorded (repetition 1, halfmove 99) the root is not claimable.
        self.assertFalse(unmodified.rows["full"].claim_draw)
        root["repetition"] = 3
        config = load_config()
        config = dataclasses.replace(
            config,
            search=dataclasses.replace(config.search, depth=CASES["rule_draw_d4"]),
        )
        claimed = synthesize_root(sidecar, root, config, seed=640)
        full = claimed.rows["full"]
        self.assertTrue(full.claim_draw)
        self.assertEqual(full.final_move, "<claim_draw>")
        replay(full.tokens, root["fen"], root["repetition"], teacher_width=1)

    def test_wide_teacher_writes_one_exact_line_per_pv_line(self):
        for name in W3_CASES:
            with self.subTest(case=name):
                sidecar, root, synthesized = synthesize_case(name, teacher_width=3)
                full = synthesized.rows["full"]
                program = replay(
                    full.tokens, root["fen"], root["repetition"], teacher_width=3
                )
                for audit, end in zip(synthesized.round_audits, sidecar.round_ends):
                    written = {candidate.move for candidate in audit.candidates}
                    for entry in end["rank"]:
                        self.assertIn(entry["move"], written)
                    self.assertGreaterEqual(len(written), 3)
                deepest = sidecar.round_ends[-1]["rank"][0]
                self.assertEqual(full.final_move, deepest["move"])
                self.assertEqual(program.final_move, deepest["move"])

    def test_wide_tie_defers_to_the_engine_rank(self):
        directory = FIXTURES / "middlegame_black_w3"
        events = [
            json.loads(line)
            for line in (directory / "sidecar.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

        def event(**fields):
            matches = [
                item
                for item in events
                if all(item.get(key) == value for key, value in fields.items())
            ]
            self.assertEqual(len(matches), 1, fields)
            return matches[0]

        # Round one ranks f6e4 (-2360) over c5f2 (-2458). Give the rank to
        # c5f2 and make the values equal: c5f2 is written after f6e4, so the
        # written values alone cannot name the champion. The engine's rank is
        # the authoritative order (docs/teacher.md); its tie-break is not
        # carried by the trace.
        promote = event(
            type="candidate", round=1, attempt=2, call="promote", move="c5f2"
        )
        promote["raw"], promote["cp"] = 2360, 624
        frontier = event(type="frontier", round=1, attempt=2, path=["c5f2"])
        frontier["raw"], frontier["cp"] = 2360, 624
        attempt = event(type="attempt", round=1, attempt=2)
        attempt["raw"], attempt["cp"] = -2360, -624
        end = event(type="round_end", round=1)
        rank = {entry["move"]: entry for entry in end["rank"]}
        rank["c5f2"].update(raw=-2360, cp=-624)
        end["rank"] = [rank["c5f2"], rank["f6e4"], rank["c6d4"]]

        sidecar = parse_sidecar(
            [json.dumps(item, separators=(",", ":")) for item in events],
            teacher_width=3,
        )
        root = json.loads((directory / "root.json").read_text(encoding="utf-8"))
        config = load_config()
        config = dataclasses.replace(
            config,
            search=dataclasses.replace(
                config.search,
                depth=W3_CASES["middlegame_black_w3"],
                teacher_width=3,
            ),
        )
        synthesized = synthesize_root(sidecar, root, config, seed=640)
        first = synthesized.round_audits[0]
        self.assertEqual(first.move, "c5f2")
        self.assertEqual(first.move, end["rank"][0]["move"])
        written = {candidate.move: candidate.raw for candidate in first.candidates}
        self.assertEqual(written["c5f2"], written["f6e4"])
        replay(
            synthesized.rows["full"].tokens,
            root["fen"],
            root["repetition"],
            teacher_width=3,
        )

    def test_wide_teacher_mate_suffix_matches_the_deepest_rank(self):
        sidecar, root, synthesized = synthesize_case("mate_in_1_w3", teacher_width=3)
        full = synthesized.rows["full"]
        deepest = sidecar.round_ends[-1]["rank"][0]
        self.assertEqual(
            full.mate_distance, mate_distance_moves(deepest["raw"], 32000, 246)
        )
        self.assertEqual(full.final_value_bin, 19 if deepest["raw"] > 0 else 0)

    def test_schema_must_match_the_teacher_width(self):
        for name in W3_CASES:
            with self.subTest(case=name):
                with self.assertRaises(SynthesisError):
                    load_case(name)
        with self.assertRaises(SynthesisError):
            load_case("start_d4", teacher_width=3)
        sidecar, root, synthesized = synthesize_case("middlegame_black_d4")
        self.assertEqual(root["turn"], "black")
        full = synthesized.rows["full"]
        program = replay(full.tokens, root["fen"], root["repetition"], teacher_width=1)
        self.assertTrue(program.mirrored)
        # Replay returns absolute moves even though the stored tokens are
        # rank-reflected; every written move must be legal in the absolute frame.
        self.assertTrue(any(step.operation == "move" for step in program.steps))

    def test_statistics_fields_are_populated(self):
        sidecar, root, synthesized = synthesize_case("start_d4")
        stats = synthesized.stats
        for name in (
            "rounds",
            "champions",
            "overturn_rate",
            "refutations",
            "refutations_per_round",
            "converged_depth",
            "root_values",
            "all_draw",
            "claim_draw",
            "thinking_tokens",
            "total_tokens",
            "board_descent",
            "board_frontier",
        ):
            self.assertIn(name, stats)
        profile = profile_roots([synthesized])
        self.assertEqual(profile["roots"], 1)
        self.assertEqual(profile["rows"], 4)
        self.assertEqual(profile["split_counts"], {"train": 1})
        self.assertEqual(profile["presentation_counts"]["full"], 1)
        self.assertGreater(profile["thinking_tokens"]["median"], 0)

    def test_seed_changes_only_presentation_and_board_choices(self):
        sidecar, root, first = synthesize_case("start_d4", seed=640)
        _, _, second = synthesize_case("start_d4", seed=641)
        self.assertEqual(
            [audit.move for audit in first.round_audits],
            [audit.move for audit in second.round_audits],
        )
        self.assertNotEqual(first.rows["full"].tokens, second.rows["full"].tokens)


class MalformedSidecarTests(unittest.TestCase):
    def read_lines(self, name):
        return (
            (FIXTURES / name / "sidecar.jsonl").read_text(encoding="utf-8").splitlines()
        )

    def renumber(self, lines):
        events = [json.loads(line) for line in lines]
        for index, event in enumerate(events, 1):
            event["seq"] = index
        return [json.dumps(event, separators=(",", ":")) for event in events]

    def test_rejects_wrong_schema_and_missing_wdl(self):
        lines = self.read_lines("start_d4")
        first = json.loads(lines[0])
        first["schema"] = 2
        with self.assertRaises(SynthesisError):
            parse_sidecar([json.dumps(first)] + lines[1:])
        events = [json.loads(line) for line in lines]
        for event in events:
            if event["type"] == "candidate" and "wdl" in event:
                del event["wdl"]
                break
        with self.assertRaises(SynthesisError):
            parse_sidecar([json.dumps(event) for event in events])

    def test_rejects_a_frontier_that_disagrees_with_its_call(self):
        lines = self.read_lines("start_d4")
        events = [json.loads(line) for line in lines]
        for event in events:
            if event["type"] == "frontier":
                event["raw"] += 1
                break
        with self.assertRaises(SynthesisError):
            parse_sidecar([json.dumps(event) for event in events])

    def test_rejects_a_root_fail_high_without_a_follow_up(self):
        lines = self.read_lines("mate_in_1_d4")
        events = [json.loads(line) for line in lines]
        # Round 1 probes e1e8 at a null window and then promotes it. Remove
        # the promote line and renumber the record.
        kept = []
        removed = False
        for event in events:
            if (
                not removed
                and event["type"] == "candidate"
                and event.get("call") == "promote"
                and event.get("move") == "e1e8"
                and event.get("round") == 1
            ):
                removed = True
                continue
            kept.append(event)
        self.assertTrue(removed)
        with self.assertRaises(SynthesisError):
            parse_sidecar(self.renumber([json.dumps(event) for event in kept]))

    def test_rejects_a_depth_zero_exact_child_without_a_frontier(self):
        lines = self.read_lines("start_d4")
        events = [json.loads(line) for line in lines]
        kept = [
            event
            for event in events
            if not (
                event["type"] == "frontier"
                and event.get("round") == 1
                and event.get("attempt") == 1
            )
        ]
        self.assertLess(len(kept), len(events))
        sidecar = parse_sidecar(self.renumber([json.dumps(event) for event in kept]))
        root = json.loads((FIXTURES / "start_d4" / "root.json").read_text())
        config = load_config()
        config = dataclasses.replace(
            config,
            search=dataclasses.replace(config.search, depth=4),
        )
        with self.assertRaises(SynthesisError):
            synthesize_root(sidecar, root, config, seed=640)


if __name__ == "__main__":
    unittest.main()
