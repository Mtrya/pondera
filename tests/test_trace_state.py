import unittest

import chess
import numpy as np

from pondera.chess.codec import decode_board, mate_token, value_bin
from pondera.chess.symmetry import TOKEN_MIRROR
from pondera.chess.tracker import PathTracker
from pondera.chess.vocabulary import (
    BACK,
    BOARD,
    DEPTH,
    END,
    FINAL,
    HALFMOVE,
    MOVE,
    PAD,
    REFUTED,
    REPETITION,
    RESET,
    SIDE,
    UCI_MOVE_TO_IDX,
    VALUE,
)
from pondera.data.trace import (
    ROLE_CONTROL,
    ROLE_EVAL,
    ROLE_FINAL_EVAL,
    ROLE_FINAL_MOVE,
    ROLE_FINAL_PV,
    ROLE_MOVE,
    TraceError,
    TraceWriter,
    replay,
)

# A black-to-move root reached after 1.e4, and a castling position with both
# sides holding all rights.
BLACK_TO_MOVE = chess.Board()
BLACK_TO_MOVE.push_uci("e2e4")
BLACK_TO_MOVE = BLACK_TO_MOVE.fen(en_passant="fen")
CASTLING_FEN = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"

# Frozen vocabulary contract: value 0 lies in bin 10, 0.5 in bin 15, and the
# extreme bins 0 and 19 are the -1 and +1 atoms.
BIN_ZERO = 10
BIN_HALF = 15
BIN_MAX = 19

# Frozen rule positions, each cross-checked against python-chess below.
FIFTY_MOVE_ROOT = "7k/8/8/8/8/8/8/K6R w - - 99 1"
FIFTY_MOVE_DRAWN_ROOT = "7k/8/8/8/8/8/8/K6R w - - 100 1"
MATE_IN_ONE_ROOT = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"


def move_token(uci):
    return MOVE + UCI_MOVE_TO_IDX[uci]


def value_token(value):
    return VALUE + value_bin(value)


def counter_trace(*thought, decision="e2e4", value=0.3, line=()):
    """A complete trace: round tokens, FINAL, value, line, and <end>."""
    suffix = [
        FINAL,
        move_token(decision),
        value_token(value),
        *[move_token(uci) for uci in line],
        END,
    ]
    return [DEPTH, *thought, *suffix]


class DesignExampleTests(unittest.TestCase):
    """The legal/illegal contrasts stated in the grammar design."""

    def test_exact_child_then_refutation_is_legal(self):
        # e2e4 eval(+0.30) RESET d2d4 <refuted> RESET
        tokens = counter_trace(
            move_token("e2e4"),
            value_token(0.30),
            RESET,
            move_token("d2d4"),
            REFUTED,
            RESET,
        )
        program = replay(tokens, chess.STARTING_FEN)
        self.assertEqual(
            [step.operation for step in program.steps[:8]],
            [
                "depth",
                "move",
                "evaluate",
                "reset",
                "move",
                "refuted",
                "reset",
                "final",
            ],
        )

    def test_refutation_without_an_exact_reference_is_illegal(self):
        # e2e4 BACK d2d4 <refuted>: the passed child is not a reference.
        tokens = [DEPTH, move_token("e2e4"), BACK, move_token("d2d4"), REFUTED]
        with self.assertRaises(TraceError) as context:
            replay(tokens, chess.STARTING_FEN)
        self.assertEqual(context.exception.tag, "refuted_without_reference")

    def test_return_evaluation_makes_the_parent_candidate_exact(self):
        # e2e4 e7e5 eval BACK c7c5 <refuted> BACK eval RESET d2d4 <refuted>
        tokens = counter_trace(
            move_token("e2e4"),
            move_token("e7e5"),
            value_token(0.28),
            BACK,
            move_token("c7c5"),
            REFUTED,
            BACK,
            value_token(0.28),
            RESET,
            move_token("d2d4"),
            REFUTED,
        )
        program = replay(tokens, chess.STARTING_FEN)
        self.assertEqual(
            [step.operation for step in program.steps[:12]],
            [
                "depth",
                "move",
                "move",
                "evaluate",
                "back",
                "move",
                "refuted",
                "back",
                "evaluate",
                "reset",
                "move",
                "refuted",
            ],
        )


class VisitStateTests(unittest.TestCase):
    def assertIllegal(self, tokens, tag, fen=chess.STARTING_FEN, **kwargs):
        with self.assertRaises(TraceError) as context:
            replay(tokens, fen, **kwargs)
        self.assertEqual(context.exception.tag, tag)

    def test_writing_a_conclusion_ends_the_visit(self):
        self.assertIllegal(
            [DEPTH, move_token("e2e4"), value_token(0.1), move_token("e7e5")],
            "move_after_conclusion",
        )
        self.assertIllegal(
            [DEPTH, move_token("e2e4"), value_token(0.1), value_token(0.2)],
            "evaluation_after_conclusion",
        )
        self.assertIllegal(
            [DEPTH, move_token("e2e4"), value_token(0.1), move_token("e7e5")],
            "move_after_conclusion",
        )
        self.assertIllegal(
            [DEPTH, move_token("e2e4"), value_token(0.1), REFUTED],
            "refuted_after_content",
        )

    def test_exits_stay_legal_after_a_conclusion(self):
        tokens = counter_trace(
            move_token("e2e4"),
            value_token(0.1),
            BACK,
            move_token("d2d4"),
            value_token(0.2),
            RESET,
        )
        replay(tokens, chess.STARTING_FEN)

        # A BOARD request after the conclusion is the one extra operation.
        writer = TraceWriter(chess.STARTING_FEN)
        writer.depth()
        writer.move("e2e4")
        writer.evaluate(value=0.1)
        writer.board()
        writer.back()
        writer.move("d2d4")
        writer.evaluate(value=0.2)
        writer.reset()
        writer.final("e2e4")
        writer.final_evaluate(value=0.1)
        writer.end()
        program = replay(writer.tokens, chess.STARTING_FEN)
        self.assertEqual(program.steps[3].operation, "board")

    def test_refutation_needs_a_fresh_visit(self):
        # The reference exists (e2e4 is exact) but d2d4 already observed.
        writer = TraceWriter(chess.STARTING_FEN)
        writer.depth()
        writer.move("e2e4")
        writer.evaluate(value=0.1)
        writer.reset()
        writer.move("d2d4")
        writer.board()
        tokens = list(writer.tokens) + [REFUTED]
        with self.assertRaises(TraceError) as context:
            replay(tokens, chess.STARTING_FEN)
        self.assertEqual(context.exception.tag, "refuted_after_content")
        self.assertIllegal(
            [
                DEPTH,
                move_token("e2e4"),
                value_token(0.1),
                RESET,
                move_token("d2d4"),
                move_token("g8f6"),
                BACK,
                REFUTED,
            ],
            "refuted_after_content",
        )

    def test_refutation_counts_other_exact_children(self):
        two_exact = [
            DEPTH,
            move_token("e2e4"),
            value_token(0.1),
            RESET,
            move_token("d2d4"),
            value_token(0.2),
            RESET,
            move_token("c2c4"),
            REFUTED,
        ]
        complete = two_exact + [FINAL, move_token("e2e4"), value_token(0.1), END]
        replay(complete, chess.STARTING_FEN, teacher_width=2)
        with self.assertRaises(TraceError) as context:
            replay(complete, chess.STARTING_FEN, teacher_width=3)
        self.assertEqual(context.exception.tag, "refuted_without_reference")

        # The current candidate's own earlier exact value is not a reference.
        self.assertIllegal(
            [
                DEPTH,
                move_token("e2e4"),
                value_token(0.1),
                RESET,
                move_token("e2e4"),
                REFUTED,
            ],
            "refuted_without_reference",
        )

    def test_a_round_root_evaluation_closes_the_round(self):
        self.assertIllegal(
            [DEPTH, move_token("e2e4"), BACK, value_token(0.2), RESET],
            "reset_after_root_evaluation",
        )
        self.assertIllegal(
            [
                DEPTH,
                move_token("e2e4"),
                BACK,
                value_token(0.2),
                move_token("d2d4"),
            ],
            "move_after_conclusion",
        )
        # <depth> and FINAL remain legal after the root evaluation.
        replay(
            counter_trace(move_token("e2e4"), BACK, value_token(0.2)),
            chess.STARTING_FEN,
        )
        replay(
            [
                DEPTH,
                move_token("e2e4"),
                BACK,
                value_token(0.2),
                DEPTH,
                FINAL,
                move_token("d2d4"),
                value_token(0.2),
                END,
            ],
            chess.STARTING_FEN,
        )

    def test_depth_rules(self):
        self.assertIllegal([move_token("e2e4")], "move_before_depth")
        self.assertIllegal([value_token(0.1)], "evaluation_before_depth")
        self.assertIllegal([DEPTH, move_token("e2e4"), DEPTH], "depth_at_depth")
        self.assertIllegal([DEPTH, DEPTH], "depth_without_candidate")
        self.assertIllegal([DEPTH, value_token(0.1), DEPTH], "depth_without_candidate")
        self.assertIllegal(
            counter_trace(move_token("e2e4"), BACK) + [DEPTH],
            "trailing_tokens",
        )

        # A passed candidate is still a candidate for the next round.
        replay(
            [
                DEPTH,
                move_token("e2e4"),
                BACK,
                DEPTH,
                FINAL,
                move_token("e2e4"),
                value_token(0.1),
                END,
            ],
            chess.STARTING_FEN,
        )
        # FINAL may immediately follow <depth> at a round-start truncation.
        replay(
            [DEPTH, FINAL, move_token("e2e4"), value_token(0.1), END],
            chess.STARTING_FEN,
        )
        # <depth> after FINAL belongs to no grammar rule.
        self.assertIllegal(
            [DEPTH, FINAL, move_token("e2e4"), value_token(0.1), DEPTH],
            "unexpected_token",
        )


class RuleTerminalTests(unittest.TestCase):
    def test_moves_stop_at_a_rule_terminal(self):
        # h1h2 makes the halfmove clock 100: the child is a 50-move draw.
        tokens = [
            DEPTH,
            move_token("h1h2"),
            value_token(0.0),
            RESET,
            FINAL,
            move_token("h1h2"),
            value_token(0.0),
            END,
        ]
        replay(tokens, FIFTY_MOVE_ROOT)
        with self.assertRaises(TraceError) as context:
            replay([DEPTH, move_token("h1h2"), move_token("h8g8")], FIFTY_MOVE_ROOT)
        self.assertEqual(context.exception.tag, "move_after_terminal")

    def test_a_terminal_root_cannot_be_expanded(self):
        with self.assertRaises(TraceError) as context:
            replay([DEPTH, move_token("h1h2")], FIFTY_MOVE_DRAWN_ROOT)
        self.assertEqual(context.exception.tag, "move_after_terminal")

    def test_checkmate_child_is_a_terminal(self):
        # f7g7 is mate; checkmate wins over the 50-move clock.
        tokens = [
            DEPTH,
            move_token("f7g7"),
            mate_token(0, True),
            BACK,
            FINAL,
            move_token("f7g7"),
            value_token(1.0),
            END,
        ]
        replay(tokens, MATE_IN_ONE_ROOT)
        with self.assertRaises(TraceError) as context:
            replay([DEPTH, move_token("f7g7"), move_token("h8g8")], MATE_IN_ONE_ROOT)
        self.assertEqual(context.exception.tag, "move_after_terminal")

    def test_insufficient_material_does_not_stop_the_trace(self):
        # K vs K is game over by material but not by the search rule.
        fen = "8/8/8/4k3/8/8/8/4K3 w - - 0 1"
        self.assertTrue(chess.Board(fen).is_game_over())
        self.assertIsNone(PathTracker(fen).search_terminal())
        writer = TraceWriter(fen)
        writer.depth()
        writer.move("e1e2")
        writer.back()
        writer.evaluate(value=0.0)
        writer.final("e1e2")
        writer.final_evaluate(value=0.0)
        writer.end()
        replay(writer.tokens, fen)


class FinalSuffixTests(unittest.TestCase):
    def assertIllegal(self, tokens, tag, fen=chess.STARTING_FEN, **kwargs):
        with self.assertRaises(TraceError) as context:
            replay(tokens, fen, **kwargs)
        self.assertEqual(context.exception.tag, tag)

    def test_normal_suffix_with_a_best_line(self):
        tokens = [
            FINAL,
            move_token("e2e4"),
            value_token(0.5),
            move_token("e7e5"),
            move_token("g1f3"),
            END,
        ]
        program = replay(tokens, chess.STARTING_FEN)
        self.assertEqual(program.final_move, "e2e4")
        self.assertEqual(program.final_move_index, 1)
        self.assertNotEqual(program.final_move_index, len(program.tokens) - 1)
        self.assertEqual(
            [step.operation for step in program.steps],
            ["final", "final_evaluate", "final_pv", "final_pv", "end"],
        )
        self.assertEqual(program.steps[2].path, ("e2e4", "e7e5"))
        self.assertEqual(
            list(program.roles),
            [
                ROLE_CONTROL,
                ROLE_FINAL_MOVE,
                ROLE_FINAL_EVAL,
                ROLE_FINAL_PV,
                ROLE_FINAL_PV,
                ROLE_CONTROL,
            ],
        )

    def test_mate_token_needs_the_matching_extreme_bin(self):
        replay(
            [FINAL, move_token("e2e4"), VALUE + BIN_MAX, mate_token(3, True), END],
            chess.STARTING_FEN,
        )
        replay(
            [FINAL, move_token("e2e4"), VALUE + 0, mate_token(4, False), END],
            chess.STARTING_FEN,
        )
        self.assertIllegal(
            [FINAL, move_token("e2e4"), VALUE + BIN_HALF, mate_token(3, True), END],
            "mate_after_non_extreme",
        )
        self.assertIllegal(
            [FINAL, move_token("e2e4"), VALUE + BIN_MAX, mate_token(3, False), END],
            "mate_after_non_extreme",
        )
        self.assertIllegal(
            [FINAL, move_token("e2e4"), VALUE + 0, mate_token(3, True), END],
            "mate_after_non_extreme",
        )

    def test_mate_token_must_immediately_follow_the_value(self):
        self.assertIllegal(
            [FINAL, move_token("e2e4"), mate_token(3, True), END],
            "final_without_value",
        )
        self.assertIllegal(
            [
                FINAL,
                move_token("e2e4"),
                VALUE + BIN_MAX,
                move_token("e7e5"),
                mate_token(3, True),
                END,
            ],
            "mate_after_line",
        )
        # A line after a mate announcement is legal.
        replay(
            [
                FINAL,
                move_token("e2e4"),
                VALUE + BIN_MAX,
                mate_token(3, True),
                move_token("e7e5"),
                END,
            ],
            chess.STARTING_FEN,
        )

    def test_suffix_shapes_are_ordered(self):
        self.assertIllegal([FINAL], "final_without_move")
        self.assertIllegal([FINAL, VALUE + BIN_ZERO], "final_without_move")
        self.assertIllegal([FINAL, move_token("e2e4")], "final_without_value")
        self.assertIllegal(
            [FINAL, move_token("e2e4"), VALUE + BIN_ZERO, VALUE + BIN_HALF, END],
            "unexpected_token",
        )
        self.assertIllegal(
            [DEPTH, FINAL, move_token("e2e4"), VALUE + BIN_ZERO], "missing_end"
        )
        self.assertIllegal([DEPTH], "missing_final")
        self.assertIllegal([END], "end_before_final")
        self.assertIllegal(
            [DEPTH, FINAL, move_token("e2e4"), VALUE + BIN_ZERO, END, END],
            "trailing_tokens",
        )

    def test_final_line_is_legal_from_the_root(self):
        self.assertIllegal(
            [
                FINAL,
                move_token("e2e4"),
                VALUE + BIN_ZERO,
                move_token("e2e4"),
                END,
            ],
            "final_illegal",
        )
        # d2d4 is legal at the root but not after e2e4.
        self.assertIllegal(
            [
                FINAL,
                move_token("e2e4"),
                VALUE + BIN_ZERO,
                move_token("d2d4"),
                END,
            ],
            "final_illegal",
        )

    def test_final_line_stops_at_a_rule_terminal(self):
        self.assertIllegal(
            [
                FINAL,
                move_token("f7g7"),
                VALUE + BIN_MAX,
                mate_token(0, True),
                move_token("h8g8"),
                END,
            ],
            "final_pv_after_terminal",
            fen=MATE_IN_ONE_ROOT,
        )
        replay(
            [
                FINAL,
                move_token("f7g7"),
                VALUE + BIN_MAX,
                mate_token(0, True),
                END,
            ],
            MATE_IN_ONE_ROOT,
        )

    def test_draw_claim_suffix(self):
        claim = [FINAL, move_token("<claim_draw>"), value_token(0.25), END]
        program = replay(claim, chess.STARTING_FEN, repetition=3)
        self.assertEqual(program.final_move, "<claim_draw>")
        self.assertEqual(program.roles[1], ROLE_FINAL_MOVE)

        self.assertIllegal(claim, "final_illegal", repetition=2)
        self.assertIllegal(
            [
                FINAL,
                move_token("<claim_draw>"),
                value_token(0.25),
                move_token("e2e4"),
                END,
            ],
            "claim_with_variation",
            repetition=3,
        )
        # A claim may carry the mate token that justifies it.
        replay(
            [FINAL, move_token("<claim_draw>"), VALUE + 0, mate_token(2, False), END],
            chess.STARTING_FEN,
            repetition=3,
        )
        self.assertIllegal(
            [
                FINAL,
                move_token("<claim_draw>"),
                VALUE + BIN_HALF,
                mate_token(2, False),
                END,
            ],
            "mate_after_non_extreme",
            repetition=3,
        )

    def test_halfmove_clock_claim_and_terminal_root(self):
        claim = [FINAL, move_token("<claim_draw>"), value_token(0.0), END]
        replay(claim, FIFTY_MOVE_DRAWN_ROOT)
        self.assertIllegal(
            [FINAL, move_token("h1h2"), value_token(0.0), END],
            "final_illegal",
            fen=FIFTY_MOVE_DRAWN_ROOT,
        )
        # The claim token is not rank-reflected with a black root.
        claim = [FINAL, move_token("<claim_draw>"), value_token(0.0), END]
        replay(claim, BLACK_TO_MOVE, repetition=3)

    def test_black_root_mirrors_the_decision_and_the_line(self):
        writer = TraceWriter(BLACK_TO_MOVE)
        writer.depth()
        writer.move("e7e5")
        writer.back()
        writer.evaluate(value=0.2)
        writer.final("c7c5")
        writer.final_evaluate(value=0.1)
        writer.final_pv("d2d4")
        writer.end()
        program = replay(writer.tokens, BLACK_TO_MOVE)
        self.assertEqual(program.final_move, "c7c5")
        self.assertEqual(program.steps[-2].path, ("c7c5", "d2d4"))
        self.assertEqual(program.steps[-2].uci, "d2d4")
        # The stored tokens are rank-reflected while the replay is absolute.
        from pondera.chess.vocabulary import IDX_TO_UCI_MOVE

        stored = [
            IDX_TO_UCI_MOVE[int(token) - MOVE]
            for token in writer.tokens
            if int(token) >= MOVE and int(token) < VALUE
        ]
        self.assertEqual(stored, ["e2e4", "c2c4", "d7d5"])


class MaskParityTests(unittest.TestCase):
    """The generation mask and the writer share the frozen trace contract."""

    def test_each_frozen_token_is_allowed_and_accepted(self):
        hand = [
            DEPTH,
            move_token("e2e4"),
            BACK,
            move_token("d2d4"),
            value_token(0.3),
            BACK,
            value_token(0.1),
            FINAL,
            move_token("d2d4"),
            value_token(0.1),
            move_token("g8f6"),
            END,
        ]
        writer = TraceWriter(chess.STARTING_FEN)
        for index, token in enumerate(hand):
            with self.subTest(index=index, token=token):
                self.assertIn(token, writer.allowed_tokens(final_allowed=True))
                writer.apply(token)
        self.assertTrue(writer.finished)
        self.assertEqual(writer.tokens.tolist(), hand)
        self.assertEqual(
            writer.roles.tolist(),
            [
                ROLE_CONTROL,
                ROLE_MOVE,
                ROLE_CONTROL,
                ROLE_MOVE,
                ROLE_EVAL,
                ROLE_CONTROL,
                ROLE_EVAL,
                ROLE_CONTROL,
                ROLE_FINAL_MOVE,
                ROLE_FINAL_EVAL,
                ROLE_FINAL_PV,
                ROLE_CONTROL,
            ],
        )

    def test_mask_states_match_the_grammar(self):
        writer = TraceWriter(chess.STARTING_FEN)
        self.assertEqual(writer.allowed_tokens(False), (DEPTH,))
        self.assertEqual(writer.allowed_tokens(True), (DEPTH, FINAL))
        writer.depth()
        allowed = writer.allowed_tokens(True)
        self.assertIn(FINAL, allowed)
        self.assertNotIn(REFUTED, allowed)
        self.assertNotIn(BACK, allowed)
        self.assertNotIn(BOARD, allowed)
        writer.move("e2e4")
        allowed = set(writer.allowed_tokens(True))
        self.assertIn(BACK, allowed)
        self.assertIn(BOARD, allowed)
        self.assertIn(FINAL, allowed)
        self.assertTrue(any(VALUE <= token < PAD for token in allowed))
        writer.board()
        # A written visit can no longer be refuted.
        self.assertNotIn(REFUTED, writer.allowed_tokens(True))
        writer.move("e7e5")
        writer.evaluate(value=0.1)
        allowed = set(writer.allowed_tokens(True))
        self.assertNotIn(move_token("g8f6"), allowed)
        self.assertEqual(
            allowed & {BACK, RESET, BOARD, FINAL}, {BACK, RESET, BOARD, FINAL}
        )

        writer = TraceWriter(chess.STARTING_FEN)
        writer.begin_final()
        allowed = set(writer.allowed_tokens(True))
        self.assertIn(move_token("d2d4"), allowed)
        self.assertNotIn(FINAL, allowed)
        writer.final_decision("e2e4")
        self.assertEqual(
            set(writer.allowed_tokens(True)), set(range(VALUE, VALUE + 20))
        )
        writer.final_evaluate(value=0.0)
        allowed = set(writer.allowed_tokens(True))
        self.assertIn(END, allowed)
        self.assertIn(move_token("e7e5"), allowed)
        self.assertNotIn(FINAL, allowed)


class BoardObservationTests(unittest.TestCase):
    def test_board_observations_match_the_true_position(self):
        writer = TraceWriter(CASTLING_FEN)
        writer.depth()
        writer.move("e1g1")
        writer.board()
        writer.final("a1b1")
        writer.final_evaluate(value=0.0)
        writer.end()
        program = replay(writer.tokens, CASTLING_FEN)
        board = chess.Board(CASTLING_FEN)
        board.push_uci("e1g1")
        step = program.steps[2]
        self.assertEqual(
            step.observed.split()[:5], board.fen(en_passant="fen").split()[:5]
        )
        self.assertEqual(step.repetition, 1)
        decoded, repetition = decode_board(
            np.asarray(writer.tokens[step.start + 1 : step.end])
        )
        self.assertEqual(decoded.split()[:5], board.fen(en_passant="fen").split()[:5])
        self.assertEqual(repetition, 1)

    def test_board_observation_saturates_large_halfmove_clock(self):
        root_fen = "7k/8/8/8/8/8/8/K6R w - - 99 1"
        writer = TraceWriter(root_fen)
        writer.depth()
        writer.move("h1h2")
        writer.board()
        writer.final("h1h2")
        writer.final_evaluate(value=0.0)
        writer.end()
        board = chess.Board(root_fen)
        board.push_uci("h1h2")
        self.assertEqual(board.halfmove_clock, 100)

        program = replay(writer.tokens, root_fen)
        step = program.steps[2]
        self.assertEqual(step.position, board.fen(en_passant="fen"))
        self.assertEqual(step.observed.split()[:4], board.fen().split()[:4])
        self.assertEqual(step.observed.split()[4], "100")
        self.assertEqual(writer.tokens[step.end - 2], HALFMOVE + 100)

    def test_board_observation_rejects_wrong_or_broken_content(self):
        writer = TraceWriter(chess.STARTING_FEN)
        writer.depth()
        writer.move("e2e4")
        writer.board()
        writer.final("d2d4")
        writer.final_evaluate(value=0.0)
        writer.end()
        step = writer.steps[2]
        tokens = list(writer.tokens)

        wrong = list(tokens)
        wrong[step.start + 1] = 0  # a1 pawn atom, not the observed position
        with self.assertRaises(TraceError) as context:
            replay(wrong, chess.STARTING_FEN)
        self.assertEqual(context.exception.tag, "board_mismatch")

        truncated = tokens[: step.end - 1]
        with self.assertRaises(TraceError) as context:
            replay(truncated, chess.STARTING_FEN)
        self.assertEqual(context.exception.tag, "board_truncated")

        invalid = list(tokens)
        invalid[step.end - 7] = SIDE  # duplicate side atom
        with self.assertRaises(TraceError) as context:
            replay(invalid, chess.STARTING_FEN)
        self.assertEqual(context.exception.tag, "board_invalid")

        # The observation carries the true halfmove clock and repetition count.
        for offset, token in (
            (step.end - 2, HALFMOVE + 7),  # a clock the position does not have
            (step.end - 1, REPETITION + 1),  # claims a second occurrence
        ):
            with self.subTest(offset=offset):
                wrong_counter = list(tokens)
                wrong_counter[offset] = token
                with self.assertRaises(TraceError) as context:
                    replay(wrong_counter, chess.STARTING_FEN)
                self.assertEqual(context.exception.tag, "board_mismatch")

    def test_board_atoms_are_a_set_not_a_byte_sequence(self):
        # Replay validates by decoding the observation, so piece atoms may
        # arrive in any order and still describe the same position.
        writer = TraceWriter(chess.STARTING_FEN)
        writer.depth()
        writer.move("e2e4")
        writer.board()
        writer.final("d2d4")
        writer.final_evaluate(value=0.0)
        writer.end()
        step = writer.steps[2]
        tokens = list(writer.tokens)
        pieces = tokens[step.start + 1 : step.end - 8]
        reordered = (
            tokens[: step.start + 1] + list(reversed(pieces)) + tokens[step.end - 8 :]
        )
        self.assertNotEqual(reordered, tokens)
        program = replay(reordered, chess.STARTING_FEN)
        self.assertEqual(program.steps[2].observed, writer.steps[2].observed)

    def test_board_at_the_root_is_illegal(self):
        tracker = PathTracker(chess.STARTING_FEN)
        tokens = [DEPTH, BOARD, *[int(token) for token in tracker.board_tokens()]]
        with self.assertRaises(TraceError) as context:
            replay(tokens, chess.STARTING_FEN)
        self.assertEqual(context.exception.tag, "board_at_root")
        writer = TraceWriter(chess.STARTING_FEN)
        writer.depth()
        with self.assertRaises(ValueError):
            writer.board()

    def test_board_response_mirrors_pieces_but_not_side_or_counters(self):
        root = chess.Board()
        for move in ("e2e4", "e7e5", "g1f3"):
            root.push_uci(move)
        root_fen = root.fen(en_passant="fen")
        self.assertEqual(root.turn, chess.BLACK)
        self.assertEqual(root.halfmove_clock, 1)

        writer = TraceWriter(root_fen)
        self.assertTrue(writer.mirrored)
        writer.depth()
        writer.move("b8c6")  # odd ply: White to move again, clock 2
        writer.board()
        writer.final("d7d5")
        writer.final_evaluate(value=0.0)
        writer.end()
        program = replay(writer.tokens, root_fen)

        true_board = chess.Board(root_fen)
        true_board.push_uci("b8c6")
        self.assertEqual(true_board.turn, chess.WHITE)
        self.assertEqual(true_board.halfmove_clock, 2)
        step = program.steps[2]
        self.assertEqual(
            step.observed.split()[:5], true_board.fen(en_passant="fen").split()[:5]
        )
        # The stored side atom names the true side, not the mirrored one.
        self.assertEqual(step.observed.split()[1], "w")

        stored = writer.tokens[step.start + 1 : step.end]
        directly, _ = decode_board(stored)
        self.assertEqual(directly.split()[1], "w")
        self.assertNotEqual(directly.split()[0], true_board.board_fen())
        absolute, _ = decode_board(np.asarray(TOKEN_MIRROR)[stored])
        self.assertEqual(
            absolute.split()[:5], true_board.fen(en_passant="fen").split()[:5]
        )
        # A naive Board.mirror() would flip the side to move; the response does not.
        self.assertEqual(true_board.mirror().fen().split()[1], "b")


class WriterBoundaryTests(unittest.TestCase):
    def test_final_uses_the_root_legal_set_not_the_current_node(self):
        writer = TraceWriter(chess.STARTING_FEN)
        writer.depth()
        writer.move("e2e4")
        writer.board()
        writer.final("d2d4")
        writer.final_evaluate(value=0.0)
        writer.end()
        program = replay(writer.tokens, chess.STARTING_FEN)
        self.assertEqual(program.final_move, "d2d4")
        self.assertEqual(program.steps[-3].path, ("e2e4",))

    def test_nothing_follows_end(self):
        writer = TraceWriter(chess.STARTING_FEN)
        writer.final("e2e4")
        writer.final_evaluate(value=0.0)
        writer.end()
        self.assertTrue(writer.finished)
        for operation in (
            writer.depth,
            lambda: writer.move("d2d4"),
            writer.back,
            writer.reset,
            writer.refuted,
            lambda: writer.evaluate(value=0.1),
            writer.board,
            lambda: writer.final("d2d4"),
            lambda: writer.final_evaluate(value=0.1),
            lambda: writer.final_pv("e7e5"),
            writer.end,
            lambda: writer.apply(END),
        ):
            with self.assertRaises(ValueError):
                operation()

    def test_claim_draw_is_not_a_thought_move(self):
        writer = TraceWriter(chess.STARTING_FEN, repetition=3)
        writer.depth()
        with self.assertRaises(ValueError):
            writer.move("<claim_draw>")
        with self.assertRaises(TraceError) as context:
            replay(
                [DEPTH, move_token("<claim_draw>")],
                chess.STARTING_FEN,
                repetition=3,
            )
        self.assertEqual(context.exception.tag, "illegal_move")

    def test_malformed_streams_report_padding_and_unexpected_tokens(self):
        with self.assertRaises(TraceError) as context:
            replay([DEPTH, PAD, FINAL], chess.STARTING_FEN)
        self.assertEqual(context.exception.tag, "pad_inside_trace")
        with self.assertRaises(TraceError) as context:
            replay([SIDE], chess.STARTING_FEN)
        self.assertEqual(context.exception.tag, "unexpected_token")


if __name__ == "__main__":
    unittest.main()
