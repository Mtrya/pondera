import unittest

from pondera.chess import _rules as rules
from pondera.chess.codec import decode_board
from pondera.chess.tracker import PathTracker

CASTLING_FEN = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"


class PathTrackerTests(unittest.TestCase):
    def test_qualified_push_preserves_san_history_and_undo(self):
        for fen, steps in (
            (
                CASTLING_FEN,
                (
                    ("e1g1", "O-O", "r3k2r/8/8/8/8/8/8/R4RK1 b kq - 1 1"),
                    ("e8c8", "O-O-O", "2kr3r/8/8/8/8/8/8/R4RK1 w - - 2 2"),
                ),
            ),
            (
                "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
                (("e5d6", "exd6", "4k3/8/3P4/8/8/8/8/4K3 b - - 0 1"),),
            ),
            (
                "7k/P7/8/8/8/8/8/4K3 w - - 0 1",
                (("a7a8q", "a8=Q+", "Q6k/8/8/8/8/8/8/4K3 b - - 0 1"),),
            ),
        ):
            with self.subTest(fen=fen):
                tracker = PathTracker(fen)
                for move, san, result in steps:
                    self.assertEqual(tracker.push_legal(move), san)
                    self.assertEqual(tracker.fen(), result)
                self.assertEqual(tracker.path, tuple(step[0] for step in steps))
                self.assertEqual(tracker.repetition(), 1)
                for _ in steps:
                    tracker.back()
                self.assertEqual(tracker.fen(), fen)
                self.assertEqual(tracker.repetition(), 1)

    def test_push_and_back_restore_position_and_moves(self):
        tracker = PathTracker(rules.STARTING_FEN)
        self.assertEqual(tracker.path, ())
        tracker.push("g1f3")
        after_nf3 = "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1"
        self.assertEqual(tracker.path, ("g1f3",))
        self.assertEqual(tracker.depth, 1)
        self.assertEqual(tracker.fen(), after_nf3)
        legal = tracker.legal_moves()
        self.assertEqual(len(legal), 20)
        self.assertIn("g8f6", legal)
        tracker.push("g8f6")
        self.assertEqual(
            tracker.fen(),
            "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2",
        )
        tracker.back()
        self.assertEqual(tracker.path, ("g1f3",))
        self.assertEqual(tracker.fen(), after_nf3)
        self.assertEqual(tracker.legal_moves(), legal)

    def test_reset_returns_to_the_root_from_any_depth(self):
        tracker = PathTracker(CASTLING_FEN)
        tracker.push("e1g1")
        tracker.push("a8b8")
        self.assertEqual(tracker.depth, 2)
        tracker.reset()
        self.assertEqual(tracker.path, ())
        self.assertEqual(tracker.depth, 0)
        self.assertEqual(tracker.fen(), CASTLING_FEN)

    def test_back_at_the_root_and_illegal_moves_raise(self):
        tracker = PathTracker(rules.STARTING_FEN)
        with self.assertRaises(ValueError):
            tracker.back()
        with self.assertRaises(ValueError):
            tracker.push("e2e5")
        with self.assertRaises(ValueError):
            tracker.push("<claim_draw>")
        self.assertEqual(tracker.path, ())

    def test_repetition_counts_complete_cycles(self):
        tracker = PathTracker(rules.STARTING_FEN)
        for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
            tracker.push(move)
        self.assertTrue(tracker.board.is_repetition(2))
        self.assertFalse(tracker.board.is_repetition(3))
        self.assertEqual(tracker.repetition(), 2)
        for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
            tracker.push(move)
        self.assertTrue(tracker.board.is_repetition(3))
        self.assertEqual(tracker.repetition(), 3)

    def test_root_repetition_joins_the_path_count(self):
        tracker = PathTracker(rules.STARTING_FEN, repetition=4)
        self.assertEqual(tracker.repetition(), 4)
        for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
            tracker.push(move)
        self.assertEqual(tracker.repetition(), 5)
        tracker.push("e2e4")
        self.assertEqual(tracker.repetition(), 1)

    def test_repetition_distinguishes_only_legal_en_passant(self):
        # The rook pins e5 to the king, making d6 an unusable ep square.
        for fen, expected in (
            ("k3r1n1/8/8/3pP3/8/8/8/4K1N1 w - d6 0 1", 2),
            ("k5n1/8/8/3pP3/8/8/8/4K1N1 w - d6 0 1", 1),
        ):
            with self.subTest(fen=fen):
                tracker = PathTracker(fen)
                for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
                    tracker.push(move)
                self.assertEqual(tracker.repetition(), expected)

    def test_repetition_history_is_restored_on_back_and_reset(self):
        tracker = PathTracker(rules.STARTING_FEN, repetition=4)
        cycle = ("g1f3", "g8f6", "f3g1", "f6g8")
        for _ in range(3):
            for move in cycle:
                tracker.push(move)
            self.assertEqual(tracker.repetition(), 5)
            for _ in cycle:
                tracker.back()
            self.assertEqual(tracker.repetition(), 4)
        for move in cycle:
            tracker.push(move)
        tracker.reset()
        self.assertEqual(tracker.repetition(), 4)
        self.assertIsNone(tracker.search_terminal())

    def test_board_tokens_are_lossless(self):
        fens = [rules.STARTING_FEN, CASTLING_FEN]
        for fen in fens:
            with self.subTest(fen=fen):
                tracker = PathTracker(fen, repetition=2)
                decoded, repetition = decode_board(tracker.board_tokens())
                self.assertEqual(
                    decoded.split()[:5],
                    fen.split()[:5],
                )
                self.assertEqual(repetition, 2)

    def test_deep_observation_keeps_en_passant_and_halfmove(self):
        tracker = PathTracker(rules.STARTING_FEN)
        tracker.push("e2e4")
        decoded, repetition = decode_board(tracker.board_tokens())
        self.assertEqual(
            decoded.split()[0], "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
        )
        self.assertEqual(decoded.split()[3], "e3")
        self.assertEqual(decoded.split()[4], "0")
        self.assertEqual(repetition, 1)


class SearchTerminalTests(unittest.TestCase):
    """Adjudication for the teacher's search path."""

    def test_root_second_occurrence_is_not_terminal(self):
        tracker = PathTracker(rules.STARTING_FEN)
        for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
            tracker.push(move)
        self.assertTrue(tracker.board.is_repetition(2))
        self.assertFalse(tracker.board.is_repetition(3))
        self.assertIsNone(tracker.search_terminal())
        self.assertEqual(tracker.repetition(), 2)

        for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
            tracker.push(move)
        self.assertTrue(tracker.board.is_repetition(3))
        self.assertEqual(tracker.search_terminal(), "draw")
        self.assertEqual(tracker.repetition(), 3)

    def test_a_position_once_after_the_root_is_terminal(self):
        tracker = PathTracker(rules.STARTING_FEN)
        for move in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"):
            tracker.push(move)
        self.assertTrue(tracker.board.is_repetition(2))
        self.assertEqual(tracker.search_terminal(), "draw")

    def test_fifty_move_rule_and_mate_priority(self):
        tracker = PathTracker("7k/8/8/8/8/8/8/K6R w - - 99 1")
        tracker.push("h1h2")
        self.assertEqual(tracker.board.halfmove_clock, 100)
        self.assertEqual(tracker.search_terminal(), "draw")

        tracker = PathTracker("7k/5Q2/6K1/8/8/8/8/8 w - - 99 1")
        tracker.push("f7g7")
        self.assertTrue(tracker.board.is_checkmate())
        self.assertEqual(tracker.board.halfmove_clock, 100)
        self.assertEqual(tracker.search_terminal(), "mate")

        # The 50-move rule is claimable at the root, so the root is searched;
        # the child reached by a non-capturing escape is a rule terminal.
        fen = "4k3/4R3/8/8/8/8/8/4K3 b - - 100 1"
        self.assertTrue(rules.Board(fen).is_check())
        self.assertFalse(rules.Board(fen).is_checkmate())
        tracker = PathTracker(fen)
        self.assertIsNone(tracker.search_terminal())
        tracker.push("e8d8")
        self.assertEqual(tracker.search_terminal(), "draw")

    def test_stalemate_and_insufficient_material(self):
        stalemate = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
        self.assertTrue(rules.Board(stalemate).is_stalemate())
        self.assertEqual(PathTracker(stalemate).search_terminal(), "draw")

        bare_kings = "8/8/8/4k3/8/8/8/4K3 w - - 0 1"
        self.assertTrue(rules.Board(bare_kings).is_game_over())
        self.assertFalse(rules.Board(bare_kings).is_stalemate())
        self.assertIsNone(PathTracker(bare_kings).search_terminal())

    def test_root_history_is_not_part_of_the_search_rule(self):
        tracker = PathTracker(rules.STARTING_FEN, repetition=5)
        self.assertEqual(tracker.repetition(), 5)
        self.assertIsNone(tracker.search_terminal())


if __name__ == "__main__":
    unittest.main()
