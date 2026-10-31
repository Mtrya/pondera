import copy
import shutil
import unittest

import chess
import chess.engine

from pondera.chess.codec import decode_value_token, value_bin
from pondera.chess.tracker import PathTracker
from pondera.chess.vocabulary import MOVE, UCI_MOVE_TO_IDX, VALUE
from pondera.data.annotation import AnnotationConfig
from pondera.data.teacher_sequence import (
    AnalysisCache,
    Candidate,
    NodeAnalysis,
    TeacherError,
    TeacherSpec,
    analyse_node,
    build_states,
    derive_states,
    emit_operations,
    materialize_state,
    plan_boards,
    restricted_value,
    root_mate,
    root_value,
    simulate_tree,
    state_for_prefix,
    synthesize_root,
    verify_teacher_trace,
    zero_thought_tokens,
)
from pondera.data.trace import replay

START = chess.STARTING_FEN


def visible(value):
    return decode_value_token(VALUE + value_bin(value))


def move_token(uci):
    return MOVE + UCI_MOVE_TO_IDX[uci]


def candidate(rank, move, value, mate=None):
    return Candidate(
        rank=rank,
        move=move,
        cp=None if mate is not None else int(round(value * 1000)),
        mate=mate,
        value=value,
        wdl=(500, 500, 0),
        pv=(move,),
    )


def make_node(tracker, d1, d4, d8):
    """A non-terminal node; each depth is a list of (move, value[, mate])."""
    legal = tuple(sorted(move.uci() for move in tracker.board.legal_moves))

    def rows(spec):
        result = []
        for rank, item in enumerate(spec, 1):
            move, value = item[0], item[1]
            mate = item[2] if len(item) > 2 else None
            if move not in legal:
                raise AssertionError(
                    f"Fixture move {move} is illegal at {tracker.path}"
                )
            result.append(candidate(rank, move, value, mate))
        return tuple(result)

    return NodeAnalysis(
        path=tracker.path,
        fen=tracker.fen(),
        repetition=tracker.repetition(),
        turn=tracker.board.turn,
        terminal=False,
        rule_value=None,
        rule_mate=None,
        rule_mate_win=None,
        rule_reason=None,
        candidates={1: rows(d1), 4: rows(d4), 8: rows(d8)},
        legal_moves=legal,
        nodes=1,
        elapsed_ms=0,
        reached_depth=8,
        reported_depth=8,
        incomplete_depths=(),
    )


def make_terminal(tracker, reason, value, mate, mate_win):
    return NodeAnalysis(
        path=tracker.path,
        fen=tracker.fen(),
        repetition=tracker.repetition(),
        turn=tracker.board.turn,
        terminal=True,
        rule_value=value,
        rule_mate=mate,
        rule_mate_win=mate_win,
        rule_reason=reason,
        candidates={},
        legal_moves=tuple(sorted(move.uci() for move in tracker.board.legal_moves)),
        nodes=None,
        elapsed_ms=0,
        reached_depth=None,
        reported_depth=None,
        incomplete_depths=(),
    )


def table_provider(table, calls):
    def provide(tracker):
        calls.append(tracker.path)
        entry = table[tracker.path]
        return entry(tracker) if callable(entry) else entry

    return provide


def auto_provider(calls, values=(0.2, 0.1)):
    """First two legal moves at every node; only shape, not chess judgment."""

    def provide(tracker):
        calls.append(tracker.path)
        moves = sorted(move.uci() for move in tracker.board.legal_moves)[:2]
        rows = [
            (move, values[0] if index == 0 else values[1])
            for index, move in enumerate(moves)
        ]
        d1 = [
            (move, value + (0.03 if index == 0 else 0.0))
            for index, (move, value) in enumerate(rows)
        ]
        return make_node(tracker, d1, rows, rows)

    return provide


def emit_tree(root_fen, repetition, tree):
    slots = plan_boards(root_fen, repetition, tree.operations)
    no_tool = emit_operations(root_fen, repetition, tree.operations, slots, tool=False)
    tool = emit_operations(root_fen, repetition, tree.operations, slots, tool=True)
    states = build_states(tree.operations, no_tool, tool)
    return slots, no_tool, tool, states


class PerspectiveTests(unittest.TestCase):
    def test_odd_depth_values_and_mates_flip_to_the_root_player(self):
        root = chess.Board()
        root.push_uci("e2e4")
        root_fen = root.fen(en_passant="fen")
        root_node = make_node(
            PathTracker(root_fen),
            [("c7c5", 0.4)],
            [("c7c5", 0.4)],
            [("c7c5", 0.4)],
        )

        def child(tracker):
            return make_node(
                tracker,
                [("g1f3", 0.3, 2)],
                [("g1f3", 0.3, 2)],
                [("g1f3", 0.3, 2)],
            )

        calls = []
        tree = simulate_tree(
            TeacherSpec("one", "fixed", max_depth=1, max_nodes=3, width=1),
            root_fen,
            1,
            table_provider({(): root_node, ("c7c5",): child}, calls),
        )
        initial = tree.operations[0]
        child_initial = tree.operations[3]
        child_leaf = tree.operations[4]
        update = tree.operations[6]
        self.assertEqual(initial["value"], visible(0.4))
        self.assertEqual(child_initial["value"], -1.0)
        self.assertEqual(child_initial["mate"], 2)
        self.assertFalse(child_initial["mate_win"])
        self.assertEqual(child_leaf["value"], -1.0)
        self.assertEqual(update["value"], -1.0)
        self.assertEqual(update["mate"], 3)
        self.assertFalse(update["mate_win"])
        self.assertEqual(tree.final_move, "c7c5")

    def test_root_helpers_flip_by_path_depth_not_search_depth(self):
        tracker = PathTracker(chess.STARTING_FEN)
        tracker.push("e2e4")
        node = make_node(
            tracker,
            [("e7e5", -0.25)],
            [("e7e5", -0.25)],
            [("e7e5", 0.5, 3)],
        )
        self.assertEqual(root_value(node, 1), 0.25)
        self.assertEqual(root_value(node, 4), 0.25)
        self.assertIsNone(root_mate(node, 1)[0])
        self.assertEqual(root_mate(node, 8), (3, False))

        root = make_node(
            PathTracker(chess.STARTING_FEN),
            [("e2e4", 0.25)],
            [("e2e4", 0.25)],
            [("e2e4", 0.5, 3)],
        )
        self.assertEqual(root_value(root, 4), 0.25)
        self.assertEqual(root_mate(root, 8), (3, True))

    def test_terminal_rule_uses_the_root_side(self):
        board = chess.Board()
        for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
            board.push_uci(move)
        fen = board.fen(en_passant="fen")
        self.assertTrue(board.is_checkmate())
        tracker = PathTracker(fen)
        white_root = analyse_node(None, tracker, AnnotationConfig(), chess.WHITE)
        self.assertTrue(white_root.terminal)
        self.assertEqual(white_root.rule_reason, "checkmate")
        self.assertEqual(white_root.rule_value, -1.0)
        self.assertEqual((white_root.rule_mate, white_root.rule_mate_win), (0, False))
        black_root = analyse_node(None, tracker, AnnotationConfig(), chess.BLACK)
        self.assertEqual(black_root.rule_value, 1.0)
        self.assertEqual((black_root.rule_mate, black_root.rule_mate_win), (0, True))

    def test_stalemate_is_a_root_draw(self):
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertTrue(board.is_stalemate())
        node = analyse_node(
            None,
            PathTracker(board.fen(en_passant="fen")),
            AnnotationConfig(),
            chess.BLACK,
        )
        self.assertTrue(node.terminal)
        self.assertEqual(node.rule_reason, "draw")
        self.assertEqual(node.rule_value, 0.0)
        self.assertIsNone(node.rule_mate)


class FixedTreeTests(unittest.TestCase):
    def test_fixed_tree_shape_matches_python_chess_branching(self):
        calls = []
        provider = auto_provider(calls)
        for spec, expected_nodes, expected_depth in (
            (TeacherSpec("d3", "fixed", max_depth=3, max_nodes=15, width=2), 15, 3),
            (TeacherSpec("d4", "fixed", max_depth=4, max_nodes=31, width=2), 31, 4),
        ):
            with self.subTest(spec=spec.name):
                calls.clear()
                tree = simulate_tree(spec, START, 1, provider)
                self.assertEqual(len(tree.nodes), expected_nodes)
                self.assertEqual(len(set(tree.nodes)), expected_nodes)
                self.assertEqual(max(len(path) for path in tree.nodes), expected_depth)
                root_moves = [
                    operation["uci"]
                    for operation in tree.operations
                    if operation["op"] == "move" and len(operation["path"]) == 1
                ]
                board = chess.Board()
                self.assertEqual(
                    root_moves,
                    sorted(move.uci() for move in board.legal_moves)[:2],
                )
                self.assertEqual(
                    len(calls), expected_nodes, "every path is its own node"
                )
                self.assertEqual(
                    tree.final_basis,
                    "completed",
                )

    def test_parent_updates_use_min_for_the_opponent_and_max_for_the_root(self):
        root_fen = START
        root = make_node(
            PathTracker(root_fen),
            [("e2e4", 0.2), ("d2d4", 0.1)],
            [("e2e4", 0.2), ("d2d4", 0.1)],
            [("e2e4", 0.2), ("d2d4", 0.1)],
        )

        def after_e4(tracker):
            return make_node(
                tracker,
                [("e7e5", -0.4), ("c7c5", -0.2)],
                [("e7e5", -0.4), ("c7c5", -0.2)],
                [("e7e5", -0.4), ("c7c5", -0.2)],
            )

        def after_d4(tracker):
            return make_node(
                tracker,
                [("d7d5", -0.6), ("g8f6", -0.1)],
                [("d7d5", -0.6), ("g8f6", -0.1)],
                [("d7d5", -0.6), ("g8f6", -0.1)],
            )

        def leaf(tracker, value):
            moves = sorted(move.uci() for move in tracker.board.legal_moves)
            return make_node(
                tracker,
                [(moves[0], value)],
                [(moves[0], value)],
                [(moves[0], value)],
            )

        table = {
            (): root,
            ("e2e4",): after_e4,
            ("d2d4",): after_d4,
            ("e2e4", "e7e5"): lambda t: leaf(t, 0.5),
            ("e2e4", "c7c5"): lambda t: leaf(t, 0.3),
            ("d2d4", "d7d5"): lambda t: leaf(t, 0.6),
            ("d2d4", "g8f6"): lambda t: leaf(t, 0.2),
        }
        calls = []
        tree = simulate_tree(
            TeacherSpec("fixed2", "fixed", max_depth=2, max_nodes=7, width=2),
            root_fen,
            1,
            table_provider(table, calls),
        )
        updates = {}
        for operation in tree.operations:
            if operation["op"] == "evaluate" and operation["source"] == "update":
                updates.setdefault(tuple(operation["path"]), []).append(
                    operation["value"]
                )
        self.assertEqual(updates[("e2e4",)], [visible(0.5), visible(0.3)])
        self.assertEqual(updates[("d2d4",)], [visible(0.6), visible(0.2)])
        self.assertEqual(updates[()], [visible(0.3), visible(0.3)])
        self.assertEqual(tree.final_move, "e2e4")
        self.assertEqual(tree.final_value, visible(0.3))

    def test_terminal_child_returns_its_rule_value(self):
        board = chess.Board()
        for move in ("f2f3", "e7e5", "g2g4"):
            board.push_uci(move)
        root_fen = board.fen(en_passant="fen")
        root = make_node(
            PathTracker(root_fen),
            [("d8h4", -1.0)],
            [("d8h4", -1.0)],
            [("d8h4", -1.0)],
        )

        def mate(tracker):
            return make_terminal(tracker, "checkmate", 1.0, 0, True)

        calls = []
        tree = simulate_tree(
            TeacherSpec("mate", "fixed", max_depth=1, max_nodes=3, width=1),
            root_fen,
            1,
            table_provider({(): root, ("d8h4",): mate}, calls),
        )
        terminal = tree.operations[3]
        self.assertEqual(terminal["source"], "terminal")
        self.assertEqual(terminal["value"], 1.0)
        self.assertEqual(terminal["mate"], 0)
        self.assertTrue(terminal["mate_win"])
        update = tree.operations[5]
        self.assertEqual(update["value"], 1.0)
        self.assertEqual(update["mate"], 1)
        self.assertTrue(update["mate_win"])
        self.assertEqual(tree.final_move, "d8h4")
        self.assertEqual(tree.final_value, 1.0)

    def test_mate_at_the_depth_limit_does_not_expand_past_the_budget(self):
        board = chess.Board()
        for move in ("f2f3", "e7e5"):
            board.push_uci(move)
        root_fen = board.fen(en_passant="fen")
        root = make_node(
            PathTracker(root_fen),
            [("g2g4", -1.0, -1)],
            [("g2g4", -1.0, -1)],
            [("g2g4", -1.0, -1)],
        )

        def child(tracker):
            return make_node(
                tracker,
                [("d8h4", 1.0, 1)],
                [("d8h4", 1.0, 1)],
                [("d8h4", 1.0, 1)],
            )

        tree = simulate_tree(
            TeacherSpec("mate-limit", "fixed", max_depth=1, max_nodes=2, width=1),
            root_fen,
            1,
            table_provider({(): root, ("g2g4",): child}, []),
        )
        self.assertEqual(len(tree.nodes), 2)
        self.assertFalse(
            any(operation.get("source") == "terminal" for operation in tree.operations)
        )
        update = next(
            operation
            for operation in tree.operations
            if operation.get("source") == "update" and operation["path"] == []
        )
        self.assertEqual(update["mate"], 2)
        self.assertFalse(update["mate_win"])
        self.assertEqual(tree.final_move, "g2g4")


class AdaptiveRuleTests(unittest.TestCase):
    def root_node(self, d1, d4, d8):
        return make_node(PathTracker(START), d1, d4, d8)

    def leaf_provider(self):
        def child(tracker):
            move = sorted(move.uci() for move in tracker.board.legal_moves)[0]
            return make_node(tracker, [(move, 0.0)], [(move, 0.0)], [(move, 0.0)])

        return child

    def run_root(self, root, spec=None):
        spec = spec or TeacherSpec("adaptive", "adaptive", max_depth=1, max_nodes=5)
        calls = []

        def provide(tracker):
            calls.append(tracker.path)
            if tracker.path == ():
                return root
            moves = sorted(move.uci() for move in tracker.board.legal_moves)
            return make_node(
                tracker,
                [(moves[0], 0.0)],
                [(moves[0], 0.0)],
                [(moves[0], 0.0)],
            )

        return simulate_tree(spec, START, 1, provide), calls

    def test_no_expansion_when_shallow_values_and_gap_are_stable(self):
        root = self.root_node(
            [("e2e4", 0.28), ("d2d4", 0.0)],
            [("e2e4", 0.29), ("d2d4", 0.0)],
            [("e2e4", 0.30), ("d2d4", 0.20)],
        )
        tree, calls = self.run_root(root)
        self.assertEqual(
            [step["op"] for step in tree.operations],
            ["evaluate", "evaluate", "final"],
        )
        self.assertEqual(tree.final_basis, "shallow")
        self.assertEqual(tree.final_move, "e2e4")
        self.assertEqual(len(calls), 1)

    def test_value_floor_is_inclusive_at_exactly_five_hundredths(self):
        root = self.root_node(
            [("e2e4", 0.30)],
            [("e2e4", 0.25)],
            [("e2e4", 0.20), ("d2d4", 0.10)],
        )
        tree, _ = self.run_root(root)
        root_moves = [
            step["uci"]
            for step in tree.operations
            if step["op"] == "move" and len(step["path"]) == 1
        ]
        self.assertEqual(root_moves, ["e2e4", "d2d4"])

    def test_gap_rule_and_candidate_order(self):
        root = self.root_node(
            [("d2d4", 0.20)],
            [("g1f3", 0.20)],
            [("e2e4", 0.20), ("b1c3", 0.15)],
        )
        tree, _ = self.run_root(root)
        root_moves = [
            step["uci"]
            for step in tree.operations
            if step["op"] == "move" and len(step["path"]) == 1
        ]
        # d8 rank 1, d4 rank 1, d1 rank 1, then the close d8 candidate.
        self.assertEqual(root_moves, ["e2e4", "g1f3", "d2d4", "b1c3"])

    def test_gap_just_above_the_floor_does_not_expand(self):
        root = self.root_node(
            [("e2e4", 0.20)],
            [("e2e4", 0.20)],
            [("e2e4", 0.20), ("d2d4", 0.14)],
        )
        tree, _ = self.run_root(root)
        self.assertEqual(tree.final_basis, "shallow")
        self.assertEqual(tree.final_move, "e2e4")

    def test_rank_one_change_expands_even_with_stable_values(self):
        root = self.root_node(
            [("d2d4", 0.20)],
            [("g1f3", 0.20)],
            [("e2e4", 0.20)],
        )
        tree, _ = self.run_root(root)
        root_moves = [
            step["uci"]
            for step in tree.operations
            if step["op"] == "move" and len(step["path"]) == 1
        ]
        self.assertEqual(root_moves, ["e2e4", "g1f3", "d2d4"])

    def test_candidate_set_stops_at_four_and_keeps_d8_rank_order(self):
        root = self.root_node(
            [("d2d4", 0.20)],
            [("g1f3", 0.20)],
            [
                ("e2e4", 0.20),
                ("b1c3", 0.19),
                ("a2a3", 0.18),
                ("h2h3", 0.17),
            ],
        )
        tree, _ = self.run_root(root)
        root_moves = [
            step["uci"]
            for step in tree.operations
            if step["op"] == "move" and len(step["path"]) == 1
        ]
        self.assertEqual(root_moves, ["e2e4", "g1f3", "d2d4", "b1c3"])

    def test_shallow_screening_spends_the_budget_on_distinct_root_candidates(self):
        calls = []

        def provider(tracker):
            calls.append(tracker.path)
            moves = sorted(move.uci() for move in tracker.board.legal_moves)[:2]
            return make_node(
                tracker,
                [(moves[0], 0.25), (moves[1], 0.10)],
                [(moves[0], 0.20), (moves[1], 0.10)],
                [(moves[0], 0.20), (moves[1], 0.10)],
            )

        spec = TeacherSpec("budget", "adaptive", max_depth=4, max_nodes=3)
        tree = simulate_tree(spec, START, 1, provider)
        self.assertEqual(len(tree.nodes), 3)
        root_moves = [
            step["uci"]
            for step in tree.operations
            if step["op"] == "move" and len(step["path"]) == 1
        ]
        self.assertEqual(root_moves, ["a2a3", "a2a4"])
        self.assertEqual(tree.final_basis, "completed")
        # Both shallow d8 values read -0.20 from the root, so the tie keeps the
        # first proposal; a2a4 would need a strictly better shallow value.
        self.assertEqual(tree.final_move, "a2a3")
        self.assertAlmostEqual(tree.final_value, visible(-0.20))

    def deep_fixture(self, max_nodes=31):
        root = self.root_node(
            [("g1f3", 0.30)],
            [("e2e4", 0.30)],
            [("e2e4", 0.30), ("d2d4", 0.25), ("g1f3", 0.05)],
        )

        def after_e4(tracker):
            return make_node(
                tracker,
                [("e7e5", -0.30)],
                [("c7c5", -0.30)],
                [("e7e5", -0.30)],
            )

        def stable(tracker, value):
            move = sorted(move.uci() for move in tracker.board.legal_moves)[0]
            return make_node(
                tracker,
                [(move, value)],
                [(move, value)],
                [(move, value)],
            )

        def after_e5(tracker):
            return make_node(
                tracker,
                [("g1f3", -0.90), ("b1c3", 0.50)],
                [("g1f3", -0.90), ("b1c3", 0.50)],
                [("g1f3", -0.90), ("b1c3", 0.50)],
            )

        def after_c5(tracker):
            return make_node(
                tracker,
                [("g1f3", -0.80), ("d2d4", 0.50)],
                [("g1f3", -0.80), ("d2d4", 0.50)],
                [("g1f3", -0.80), ("d2d4", 0.50)],
            )

        table = {
            (): root,
            ("e2e4",): after_e4,
            ("g1f3",): lambda tracker: stable(tracker, -0.05),
            ("d2d4",): lambda tracker: stable(tracker, -0.25),
            ("e2e4", "e7e5"): after_e5,
            ("e2e4", "c7c5"): after_c5,
        }
        calls = []
        records = {}

        def provide(tracker):
            calls.append(tracker.path)
            entry = table[tracker.path]
            node = entry(tracker) if callable(entry) else entry
            records[tracker.path] = node.to_record()
            return node

        spec = TeacherSpec("deep", "adaptive", max_depth=4, max_nodes=max_nodes)
        tree = simulate_tree(spec, START, 1, provide)
        return tree, calls, records

    def test_all_root_candidates_are_screened_before_the_first_deep_entry(self):
        tree, calls, _ = self.deep_fixture()
        operations = tree.operations
        first_deep = next(
            index
            for index, operation in enumerate(operations)
            if operation.get("revisit")
        )
        self.assertEqual(first_deep, 17)
        self.assertEqual(operations[first_deep]["path"], ["e2e4"])
        for operation in operations[:first_deep]:
            if operation["op"] == "move":
                self.assertEqual(len(operation["path"]), 1)

        shallow_moves = [
            operation["uci"]
            for operation in operations[1:first_deep]
            if operation["op"] == "move"
        ]
        self.assertEqual(shallow_moves, ["e2e4", "g1f3", "d2d4"])
        initial_paths = [
            tuple(operation["path"])
            for operation in operations[:first_deep]
            if operation["op"] == "evaluate" and operation["source"] == "initial"
        ]
        leaf_paths = [
            tuple(operation["path"])
            for operation in operations[:first_deep]
            if operation["op"] == "evaluate" and operation["source"] == "leaf"
        ]
        self.assertEqual(len(initial_paths), 4)
        self.assertEqual(leaf_paths, [(), ("e2e4",), ("g1f3",), ("d2d4",)])
        root_updates = [
            operation["completed"]
            for operation in operations[:first_deep]
            if operation["op"] == "evaluate"
            and operation["source"] == "update"
            and operation["path"] == []
        ]
        self.assertEqual(
            [[entry["move"] for entry in update] for update in root_updates],
            [["e2e4"], ["e2e4", "g1f3"], ["e2e4", "g1f3", "d2d4"]],
        )
        self.assertEqual(len(calls), 8)

    def test_deep_entry_replaces_the_shallow_root_value_once(self):
        tree, _, _ = self.deep_fixture()
        root_updates = [
            operation["completed"]
            for operation in tree.operations
            if operation["op"] == "evaluate"
            and operation["source"] == "update"
            and operation["path"] == []
        ]
        self.assertEqual(len(root_updates), 5)
        for update in root_updates:
            moves = [entry["move"] for entry in update]
            self.assertEqual(moves, ["e2e4", "g1f3", "d2d4"][: len(moves)])
        final_update = root_updates[-1]
        self.assertEqual(
            [(entry["move"], entry["value"]) for entry in final_update],
            [("e2e4", visible(-0.9)), ("g1f3", visible(0.05)), ("d2d4", visible(0.25))],
        )
        self.assertEqual(tree.final_move, "d2d4")
        self.assertAlmostEqual(tree.final_value, visible(0.25))
        self.assertEqual(tree.final_basis, "completed")
        final = tree.operations[-1]
        self.assertEqual(final["uci"], "d2d4")
        self.assertEqual(final["completed"], final_update)
        candidates = {entry["move"]: entry for entry in final["candidates"]}
        self.assertTrue(all(entry["completed"] for entry in candidates.values()))
        self.assertEqual(
            {move: entry["value"] for move, entry in candidates.items()},
            {"e2e4": visible(-0.9), "g1f3": visible(0.05), "d2d4": visible(0.25)},
        )

    def test_revisits_do_not_consume_nodes_or_repeat_engine_work(self):
        tree, calls, _ = self.deep_fixture(max_nodes=6)
        self.assertEqual(len(tree.nodes), 6)
        self.assertEqual(len(set(tree.nodes)), 6)
        self.assertEqual(len(calls), 6)
        self.assertEqual(len(set(calls)), 6)
        self.assertEqual(set(calls), set(tree.nodes))
        self.assertEqual(calls.count(("e2e4",)), 1)
        reentries = [
            operation
            for operation in tree.operations
            if operation["op"] == "move" and operation["path"] == ["e2e4"]
        ]
        self.assertEqual(len(reentries), 2)
        self.assertTrue(reentries[0].get("revisit", False) is False)
        self.assertTrue(reentries[1]["revisit"])
        self.assertIn(("e2e4",), tree.expanded)

    def test_budget_exhaustion_keeps_the_shallow_root_values(self):
        tree, calls, _ = self.deep_fixture(max_nodes=4)
        self.assertFalse(any(operation.get("revisit") for operation in tree.operations))
        self.assertEqual(tree.final_move, "e2e4")
        self.assertAlmostEqual(tree.final_value, visible(0.30))
        final = tree.operations[-1]
        self.assertEqual(
            [(entry["move"], entry["value"]) for entry in final["completed"]],
            [("e2e4", visible(0.30)), ("g1f3", visible(0.05)), ("d2d4", visible(0.25))],
        )
        self.assertEqual(len(calls), 4)

    def test_terminal_root_candidate_is_shallow_and_never_deepened(self):
        root_fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
        root = make_node(
            PathTracker(root_fen),
            [("a1a7", 0.10)],
            [("a1a8", 0.90)],
            [("a1a8", 0.90), ("a1a7", 0.20)],
        )

        def mate(tracker):
            return make_terminal(tracker, "checkmate", 1.0, 0, True)

        def after_ra7(tracker):
            return make_node(
                tracker,
                [("f7f6", 0.10)],
                [("f7f6", 0.10)],
                [("f7f6", 0.10)],
            )

        calls = []
        tree = simulate_tree(
            TeacherSpec("mate", "adaptive", max_depth=4, max_nodes=31),
            root_fen,
            1,
            table_provider({(): root, ("a1a8",): mate, ("a1a7",): after_ra7}, calls),
        )
        terminal_ops = [
            operation
            for operation in tree.operations
            if operation["op"] == "evaluate" and operation["source"] == "terminal"
        ]
        self.assertEqual(len(terminal_ops), 1)
        self.assertEqual(terminal_ops[0]["path"], ["a1a8"])
        self.assertEqual(terminal_ops[0]["value"], 1.0)
        for source in ("initial", "leaf"):
            self.assertFalse(
                any(
                    operation["op"] == "evaluate"
                    and operation["source"] == source
                    and operation["path"] == ["a1a8"]
                    for operation in tree.operations
                )
            )
        self.assertFalse(any(operation.get("revisit") for operation in tree.operations))
        self.assertEqual(tree.final_move, "a1a8")
        self.assertAlmostEqual(tree.final_value, 1.0)
        self.assertEqual(len(calls), 2)

    def test_truncation_states_keep_the_shallow_winner_until_deep_replacement(self):
        tree, _, _ = self.deep_fixture()
        _, _, _, states = emit_tree(START, 1, tree)
        shallow_state = next(
            state for state in states if state["op"] == 15 and not state["board"]
        )
        self.assertEqual(shallow_state["best"], "e2e4")
        self.assertEqual(shallow_state["basis"], "completed")
        self.assertAlmostEqual(shallow_state["value"], visible(0.30))
        writer = materialize_state(START, 1, tree.operations, shallow_state, tool=True)
        self.assertEqual(replay(writer.tokens, START, 1).final_move, "e2e4")
        deep_state = states[-1]
        self.assertEqual(deep_state["best"], "d2d4")
        self.assertAlmostEqual(deep_state["value"], visible(0.25))
        writer = materialize_state(START, 1, tree.operations, deep_state, tool=True)
        self.assertEqual(replay(writer.tokens, START, 1).final_move, "d2d4")

    def test_verifier_rejects_revisit_and_root_replacement_corruption(self):
        tree, _, records = self.deep_fixture()
        slots, no_tool, tool, states = emit_tree(START, 1, tree)
        record_list = list(records.values())
        verify_teacher_trace(
            START, 1, tree.operations, record_list, no_tool, tool, states, tree.spec
        )

        operations = copy.deepcopy(tree.operations)
        operations[2]["revisit"] = True
        with self.assertRaises(TeacherError):
            verify_teacher_trace(
                START, 1, operations, record_list, no_tool, tool, states, tree.spec
            )

        operations = copy.deepcopy(tree.operations)
        for operation in operations:
            if operation.get("revisit"):
                del operation["revisit"]
                break
        with self.assertRaises(TeacherError):
            verify_teacher_trace(
                START, 1, operations, record_list, no_tool, tool, states, tree.spec
            )

        operations = copy.deepcopy(tree.operations)
        root_updates = [
            operation
            for operation in operations
            if operation["op"] == "evaluate"
            and operation["source"] == "update"
            and operation["path"] == []
        ]
        root_updates[-1]["completed"][0]["value"] = -0.5
        with self.assertRaises(TeacherError):
            verify_teacher_trace(
                START, 1, operations, record_list, no_tool, tool, states, tree.spec
            )

        operations = copy.deepcopy(tree.operations)
        root_updates = [
            operation
            for operation in operations
            if operation["op"] == "evaluate"
            and operation["source"] == "update"
            and operation["path"] == []
        ]
        root_updates[-1]["completed"].append(root_updates[-1]["completed"][1])
        with self.assertRaises(TeacherError):
            verify_teacher_trace(
                START, 1, operations, record_list, no_tool, tool, states, tree.spec
            )


class EmissionTests(unittest.TestCase):
    def test_tool_copy_differs_only_in_board_requests_and_responses(self):
        calls = []
        tree = simulate_tree(
            TeacherSpec("emit", "fixed", max_depth=1, max_nodes=3, width=2),
            START,
            1,
            auto_provider(calls),
        )
        slots, no_tool, tool, states = emit_tree(START, 1, tree)
        # Both children are entered and left immediately, so the root was never
        # stale and the tool copy is identical to the no-tool copy.
        self.assertEqual(len(slots), 0)
        self.assertEqual(tool.tokens, no_tool.tokens)
        self.assertEqual(
            len(tool.tokens) - len(no_tool.tokens),
            sum(end - start for _, start, end, _ in tool.board_spans),
        )
        stripped = list(tool.tokens)
        stripped_roles = list(tool.roles)
        for _, start, end, _ in reversed(tool.board_spans):
            del stripped[start:end]
            del stripped_roles[start:end]
        self.assertEqual(tuple(stripped), no_tool.tokens)
        self.assertEqual(tuple(stripped_roles), no_tool.roles)
        program_no = replay(no_tool.tokens, START, 1)
        program_tool = replay(tool.tokens, START, 1)
        self.assertEqual(program_no.final_move, program_tool.final_move)
        self.assertIn(program_tool.final_move, program_tool.render())
        self.assertEqual(program_tool.steps[-1].operation, "final")
        self.assertTrue(states)

    def test_black_root_trace_is_mirrored_and_legal(self):
        board = chess.Board()
        board.push_uci("e2e4")
        root_fen = board.fen(en_passant="fen")
        calls = []
        tree = simulate_tree(
            TeacherSpec("emit", "fixed", max_depth=1, max_nodes=3, width=2),
            root_fen,
            1,
            auto_provider(calls),
        )
        slots, no_tool, tool, states = emit_tree(root_fen, 1, tree)
        program = replay(tool.tokens, root_fen, 1)
        self.assertTrue(program.mirrored)
        rendered = program.render()
        self.assertIn(root_fen, rendered)
        self.assertIn(program.final_move, rendered)
        self.assertEqual(program.final_move, tree.final_move)
        self.assertEqual(
            len(states),
            len(derive_states(tree.operations)) + len(tool.board_spans),
        )

    def test_sixteen_unobserved_plies_trigger_one_board_before_expansion(self):
        operations = [
            {
                "op": "evaluate",
                "source": "initial",
                "path": [],
                "d": 1,
                "value": 0.0,
                "mate": None,
                "mate_win": None,
                "rank1": "e2e4",
            }
        ]
        moves = [
            "e2e4",
            "e7e5",
            "g1f3",
            "b8c6",
            "f1c4",
            "g8f6",
            "e1g1",
            "f8c5",
            "d2d3",
            "d7d6",
            "c1e3",
            "e8g8",
            "b1c3",
            "c8e6",
            "f1e1",
            "f8e8",
            "a2a3",
        ]
        board = chess.Board()
        path = []
        for move in moves:
            path.append(move)
            board.push_uci(move)
            operations.append({"op": "move", "uci": move, "path": list(path)})
        operations.append({"op": "final", "uci": "e2e4", "path": list(path)})
        slots = plan_boards(START, 1, operations)
        self.assertEqual(slots, (17,))
        tool = emit_operations(START, 1, operations, slots, tool=True)
        no_tool = emit_operations(START, 1, operations, slots, tool=False)
        self.assertEqual(len(tool.board_spans), 1)
        op_index, _, _, _ = tool.board_spans[0]
        self.assertEqual(op_index, 17)
        program = replay(tool.tokens, START, 1)
        board_step = next(step for step in program.steps if step.operation == "board")
        self.assertEqual(board_step.path, tuple(moves[:16]))
        board = chess.Board()
        for move in moves[:16]:
            board.push_uci(move)
        self.assertEqual(
            board_step.observed.split()[:5],
            board.fen(en_passant="fen").split()[:5],
        )
        self.assertEqual(
            len(tool.tokens) - len(no_tool.tokens),
            tool.board_spans[0][2] - tool.board_spans[0][1],
        )

    @staticmethod
    def excursion_operations(moves, evaluate=False, next_move=True):
        operations = [
            {
                "op": "evaluate",
                "source": "initial",
                "path": [],
                "d": 1,
                "value": 0.0,
                "mate": None,
                "mate_win": None,
                "rank1": "e2e4",
            }
        ]
        path = []
        for move in moves:
            path.append(move)
            operations.append({"op": "move", "uci": move, "path": list(path)})
        for depth in range(len(moves) - 1, 0, -1):
            operations.append({"op": "back", "path": list(path[:depth])})
        if evaluate:
            operations.append(
                {
                    "op": "evaluate",
                    "source": "initial",
                    "path": path[:1],
                    "d": 1,
                    "value": 0.0,
                    "mate": None,
                    "mate_win": None,
                    "rank1": "e7e5",
                }
            )
        if next_move:
            operations.append(
                {"op": "move", "uci": "e7e5", "path": path[:1] + ["e7e5"]}
            )
        operations.append({"op": "final", "uci": "e2e4", "path": list(path[:1])})
        return operations

    def test_return_after_three_plies_does_not_request_board(self):
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
        operations = self.excursion_operations(moves)
        self.assertEqual(plan_boards(START, 1, operations), ())

    def test_return_after_seven_plies_does_not_request_board(self):
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1", "f8c5", "d2d3"]
        operations = self.excursion_operations(moves, evaluate=True)
        self.assertEqual(plan_boards(START, 1, operations), ())

    def test_return_after_eight_plies_requests_board_before_the_next_move(self):
        moves = [
            "e2e4",
            "e7e5",
            "g1f3",
            "b8c6",
            "f1c4",
            "g8f6",
            "e1g1",
            "f8c5",
            "d2d3",
            "d7d6",
        ]
        operations = self.excursion_operations(moves, evaluate=True)
        # The first candidate was left at move 2 and the chain ends at move 10,
        # so its age is eight; the evaluation in between does not clear it.
        self.assertEqual(plan_boards(START, 1, operations), (21,))

    def test_revisit_is_location_and_restarts_departure_clock(self):
        operations = [
            {"op": "move", "uci": "e2e4", "path": ["e2e4"]},
            {"op": "move", "uci": "e7e5", "path": ["e2e4", "e7e5"]},
            {"op": "move", "uci": "g1f3", "path": ["e2e4", "e7e5", "g1f3"]},
            {"op": "back", "path": ["e2e4", "e7e5"]},
            {
                "op": "move",
                "uci": "g1f3",
                "path": ["e2e4", "e7e5", "g1f3"],
                "revisit": True,
            },
            {
                "op": "move",
                "uci": "b8c6",
                "path": ["e2e4", "e7e5", "g1f3", "b8c6"],
            },
            {"op": "final", "uci": "e2e4", "path": []},
        ]
        self.assertEqual(plan_boards(START, 1, operations), ())

    def test_pending_return_does_not_request_board_without_a_new_move(self):
        moves = [
            "e2e4",
            "e7e5",
            "g1f3",
            "b8c6",
            "f1c4",
            "g8f6",
            "e1g1",
            "f8c5",
            "d2d3",
            "d7d6",
        ]
        operations = self.excursion_operations(moves, evaluate=True, next_move=False)
        self.assertEqual(plan_boards(START, 1, operations), ())

    def test_root_never_requests_board_after_eight_plies_away(self):
        moves = [
            "e2e4",
            "e7e5",
            "g1f3",
            "b8c6",
            "f1c4",
            "g8f6",
            "e1g1",
            "f8c5",
            "d2d3",
            "d7d6",
        ]
        operations = [
            {
                "op": "evaluate",
                "source": "initial",
                "path": [],
                "d": 1,
                "value": 0.0,
                "mate": None,
                "mate_win": None,
                "rank1": "e2e4",
            }
        ]
        path = []
        for move in moves:
            path.append(move)
            operations.append({"op": "move", "uci": move, "path": list(path)})
        for depth in range(len(moves) - 1, 0, -1):
            operations.append({"op": "back", "path": list(path[:depth])})
        operations.append({"op": "reset", "path": []})
        operations.append({"op": "move", "uci": "d2d4", "path": ["d2d4"]})
        operations.append({"op": "final", "uci": "d2d4", "path": ["d2d4"]})
        # The root has been away for ten moves, but its board prefix is always
        # present, so the second root expansion needs no observation.
        self.assertEqual(plan_boards(START, 1, operations), ())

    def test_root_suppression_keeps_the_unobserved_ply_counter_for_non_root(self):
        legal = sorted(move.uci() for move in chess.Board().legal_moves)
        operations = [
            {
                "op": "evaluate",
                "source": "initial",
                "path": [],
                "d": 1,
                "value": 0.0,
                "mate": None,
                "mate_win": None,
                "rank1": "e2e4",
            }
        ]
        for move in legal[:17]:
            operations.append({"op": "move", "uci": move, "path": [move]})
            operations.append({"op": "reset", "path": []})
        self.assertEqual(
            plan_boards(START, 1, operations + [{"op": "final", "uci": "e2e4"}]),
            (),
        )
        operations.append({"op": "move", "uci": "a2a3", "path": ["a2a3"]})
        operations.append({"op": "move", "uci": "a7a6", "path": ["a2a3", "a7a6"]})
        operations.append({"op": "final", "uci": "a2a3", "path": ["a2a3"]})
        # Seventeen root expansions are all suppressed and do not reset the
        # counter; the next non-root expansion is the first observation.
        self.assertEqual(plan_boards(START, 1, operations), (36,))

    def test_no_board_when_the_next_operation_is_not_an_expansion(self):
        operations = [
            {
                "op": "evaluate",
                "source": "initial",
                "path": [],
                "d": 1,
                "value": 0.0,
                "mate": None,
                "mate_win": None,
                "rank1": "e2e4",
            },
            {"op": "move", "uci": "e2e4", "path": ["e2e4"]},
            {
                "op": "evaluate",
                "source": "initial",
                "path": ["e2e4"],
                "d": 1,
                "value": 0.0,
                "mate": None,
                "mate_win": None,
                "rank1": "e7e5",
            },
            {
                "op": "evaluate",
                "source": "leaf",
                "path": ["e2e4"],
                "d": 8,
                "value": 0.0,
                "mate": None,
                "mate_win": None,
                "rank1": "e7e5",
            },
            {"op": "reset", "path": []},
            {
                "op": "evaluate",
                "source": "update",
                "path": [],
                "value": 0.0,
                "completed": [{"move": "e2e4", "value": 0.0}],
                "chosen": "e2e4",
            },
            {"op": "final", "uci": "e2e4", "path": []},
        ]
        self.assertEqual(plan_boards(START, 1, operations), ())


class TruncationTests(unittest.TestCase):
    def build(self):
        root_fen = START
        root = make_node(
            PathTracker(root_fen),
            [("e2e4", 0.2), ("d2d4", 0.1)],
            [("e2e4", 0.2), ("d2d4", 0.1)],
            [("e2e4", 0.2), ("d2d4", 0.1)],
        )

        def after_e4(tracker):
            return make_node(
                tracker,
                [("e7e5", -0.1)],
                [("e7e5", -0.1)],
                [("e7e5", -0.1)],
            )

        def after_d4(tracker):
            return make_node(
                tracker,
                [("d7d5", -0.9)],
                [("d7d5", -0.9)],
                [("d7d5", -0.9)],
            )

        calls = []
        tree = simulate_tree(
            TeacherSpec("truncate", "fixed", max_depth=1, max_nodes=3, width=2),
            root_fen,
            1,
            table_provider(
                {
                    (): root,
                    ("e2e4",): after_e4,
                    ("d2d4",): after_d4,
                },
                calls,
            ),
        )
        slots, no_tool, tool, states = emit_tree(root_fen, 1, tree)
        records = [
            make_node(
                PathTracker(root_fen),
                [("e2e4", 0.2), ("d2d4", 0.1)],
                [("e2e4", 0.2), ("d2d4", 0.1)],
                [("e2e4", 0.2), ("d2d4", 0.1)],
            ).to_record()
        ]
        for path, fn in (
            (("e2e4",), after_e4),
            (("d2d4",), after_d4),
        ):
            tracker = PathTracker(root_fen)
            for move in path:
                tracker.push(move)
            records.append(fn(tracker).to_record())
        return root_fen, tree, slots, no_tool, tool, states, records

    def test_root_state_changes_only_after_the_parent_update(self):
        _, tree, _, no_tool, tool, states, _ = self.build()
        self.assertEqual(tree.final_move, "d2d4")
        self.assertEqual(tree.final_value, visible(0.9))
        updates = [
            (index, state)
            for index, state in enumerate(states)
            if state["best"] == "d2d4"
        ]
        self.assertTrue(updates)
        first_d2 = updates[0]
        # The candidate is proposed before it completes, but the best root
        # move stays the first candidate until the d2d4 child update.
        before = [state for state in states if state["tool"] < first_d2[1]["tool"]]
        self.assertTrue(any(state["best"] == "e2e4" for state in before))
        self.assertTrue(
            any(
                state["best"] == "e2e4" and state["basis"] == "proposed"
                for state in before
            )
        )

    def test_every_materialized_truncation_replays_and_keeps_its_answer(self):
        root_fen, tree, _, no_tool, tool, states, _ = self.build()
        low = max(1, int(len(tool.tokens) * 0.05))
        high = max(low, int(len(tool.tokens) * 0.95))
        materialized = 0
        for position in range(low, high + 1):
            state = state_for_prefix(states, position, tool=True)
            self.assertIsNotNone(state)
            for _, start, end, _ in tool.board_spans:
                self.assertFalse(start < state["tool"] < end)
            writer = materialize_state(root_fen, 1, tree.operations, state, tool=True)
            self.assertEqual(len(writer.tokens), state["tool"] + 2)
            truncated = replay(writer.tokens, root_fen, 1)
            self.assertEqual(truncated.final_move, state["best"])
            self.assertIn(
                truncated.final_move,
                {move.uci() for move in chess.Board(root_fen).legal_moves},
            )
            no_tool_writer = materialize_state(
                root_fen, 1, tree.operations, state, tool=False
            )
            no_tool_truncated = replay(no_tool_writer.tokens, root_fen, 1)
            self.assertEqual(no_tool_truncated.final_move, state["best"])
            self.assertEqual(len(no_tool_writer.tokens), state["no_tool"] + 2)
            materialized += 1
        self.assertGreater(materialized, 10)
        self.assertTrue(any(state["best"] == "e2e4" for state in states))
        self.assertTrue(any(state["best"] == "d2d4" for state in states))

    def test_positive_prefix_does_not_use_the_future_final_answer(self):
        root_fen, tree, _, _, _, states, _ = self.build()
        # Pick a state after the first candidate completed but before the
        # second candidate's root update. The final teacher answer is d2d4.
        target = None
        for state in states:
            if (
                state["basis"] == "completed"
                and state["best"] == "e2e4"
                and state["value"] == visible(0.1)
            ):
                target = state
        self.assertIsNotNone(target)
        writer = materialize_state(root_fen, 1, tree.operations, target, tool=True)
        program = replay(writer.tokens, root_fen, 1)
        self.assertEqual(program.final_move, "e2e4")
        self.assertNotEqual(program.final_move, tree.final_move)

    def test_zero_thought_uses_the_full_teacher_answer(self):
        root_fen, tree, _, _, _, _, _ = self.build()
        tokens = zero_thought_tokens(root_fen, 1, tree.final_move)
        program = replay(tokens, root_fen, 1)
        self.assertEqual(program.final_move, "d2d4")
        self.assertEqual(len(tokens), 2)

    def test_verifier_rejects_corrupted_operations_and_tokens(self):
        root_fen, tree, _, no_tool, tool, states, records = self.build()
        verify_teacher_trace(
            root_fen, 1, tree.operations, records, no_tool, tool, states, tree.spec
        )

        operations = copy.deepcopy(tree.operations)
        operations[-1]["uci"] = "e2e4"
        with self.assertRaises(TeacherError):
            verify_teacher_trace(
                root_fen, 1, operations, records, no_tool, tool, states, tree.spec
            )

        operations = copy.deepcopy(tree.operations)
        for operation in operations:
            if operation["op"] == "evaluate" and operation["source"] == "update":
                operation["completed"] = []
                break
        with self.assertRaises(TeacherError):
            verify_teacher_trace(
                root_fen, 1, operations, records, no_tool, tool, states, tree.spec
            )

        corrupted = list(tool.tokens)
        corrupted[0] = (
            corrupted[0] + 1 if corrupted[0] + 1 < VALUE else corrupted[0] - 1
        )
        bad_tool = type(tool)(
            tokens=tuple(corrupted),
            roles=tool.roles,
            steps=tool.steps,
            op_spans=tool.op_spans,
            board_spans=tool.board_spans,
            step_index=tool.step_index,
            writer=tool.writer,
        )
        with self.assertRaises(ValueError):
            verify_teacher_trace(
                root_fen,
                1,
                tree.operations,
                records,
                no_tool,
                bad_tool,
                states,
                tree.spec,
            )

        bad_states = copy.deepcopy(states)
        bad_states[-1]["best"] = "e2e4"
        with self.assertRaises(TeacherError):
            verify_teacher_trace(
                root_fen,
                1,
                tree.operations,
                records,
                no_tool,
                tool,
                bad_states,
                tree.spec,
            )


@unittest.skipUnless(shutil.which("stockfish"), "Requires Stockfish on PATH")
class EngineIntegrationTests(unittest.TestCase):
    def test_one_root_synthesis_replays_and_verifies(self):
        root_fen = START
        config = AnnotationConfig(
            engine=shutil.which("stockfish"),
            depth=8,
            multipv=4,
            threads=1,
            hash_mb=64,
            max_time=None,
        )
        root = {
            "root_id": "integration",
            "split": "val",
            "fen": root_fen,
            "repetition": 1,
            "source": {},
            "d16": {"bm": "e2e4", "value": 0.1, "mate": None},
        }
        with chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish")) as engine:
            engine.configure({"Threads": 1, "Hash": 64})
            record = synthesize_root(engine, root, config)
        self.assertEqual(len(record["nodes"]), record["analysis"]["nodes"])
        self.assertGreater(record["analysis"]["calls"], 0)
        for name, teacher in record["teachers"].items():
            self.assertEqual(teacher["final_move"], teacher["operations"][-1]["uci"])
            program = replay(
                teacher["tool"]["tokens"],
                root_fen,
                record["repetition"],
            )
            self.assertEqual(program.final_move, teacher["final_move"])
            self.assertLessEqual(teacher["metrics"]["nodes"], 31)
            self.assertEqual(
                teacher["verification"]["states"],
                len(teacher["truncation"]["states"]),
            )

    def test_cache_keys_nodes_by_path_not_fen(self):
        config = AnnotationConfig(
            engine=shutil.which("stockfish"),
            depth=8,
            multipv=1,
            threads=1,
            hash_mb=16,
            max_time=None,
        )
        with chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish")) as engine:
            engine.configure({"Threads": 1, "Hash": 16})
            cache = AnalysisCache(engine, config, chess.WHITE)
            first = PathTracker(START)
            for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
                first.push(move)
            second = PathTracker(START)
            for move in ("g1h3", "g8f6", "h3g1", "f6g8"):
                second.push(move)
            self.assertEqual(first.fen(), second.fen())
            node_a = cache(first)
            node_b = cache(second)
            self.assertNotEqual(node_a.path, node_b.path)
            self.assertEqual(cache.engine_calls, 2)
            self.assertEqual(len(cache), 2)

    def test_restricted_search_reports_a_forced_mate(self):
        board = chess.Board("k7/8/1K6/8/8/8/8/7R w - - 0 1")
        with chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish")) as engine:
            engine.configure({"Threads": 1, "Hash": 16})
            result = restricted_value(engine, board, "h1h8", 4)
        self.assertEqual(result["move"], "h1h8")
        self.assertGreaterEqual(result["depth"], 4)
        self.assertEqual(result["value"], 1.0)
        self.assertEqual(result["mate"], 1)
        self.assertEqual(sum(result["wdl"]), 1000)


if __name__ == "__main__":
    unittest.main()
