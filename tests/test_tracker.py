import unittest

import chess

from pondera.chess.codec import decode_board
from pondera.chess.tracker import PathTracker

CASTLING_FEN = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"


class PathTrackerTests(unittest.TestCase):
    def test_push_and_back_match_python_chess(self):
        tracker = PathTracker(chess.STARTING_FEN)
        self.assertEqual(tracker.path, ())
        self.assertEqual(
            tracker.legal_moves(),
            sorted(move.uci() for move in chess.Board().legal_moves),
        )
        tracker.push("g1f3")
        board = chess.Board()
        board.push_uci("g1f3")
        self.assertEqual(tracker.path, ("g1f3",))
        self.assertEqual(tracker.depth, 1)
        self.assertEqual(tracker.fen(), board.fen(en_passant="fen"))
        self.assertEqual(
            tracker.legal_moves(), sorted(move.uci() for move in board.legal_moves)
        )
        tracker.push("g8f6")
        board.push_uci("g8f6")
        self.assertEqual(tracker.fen(), board.fen(en_passant="fen"))
        tracker.back()
        self.assertEqual(tracker.path, ("g1f3",))
        board.pop()
        self.assertEqual(tracker.fen(), board.fen(en_passant="fen"))

    def test_reset_returns_to_the_root_from_any_depth(self):
        tracker = PathTracker(CASTLING_FEN)
        tracker.push("e1g1")
        tracker.push("a8b8")
        self.assertEqual(tracker.depth, 2)
        tracker.reset()
        self.assertEqual(tracker.path, ())
        self.assertEqual(tracker.depth, 0)
        self.assertEqual(tracker.fen(), chess.Board(CASTLING_FEN).fen(en_passant="fen"))

    def test_back_at_the_root_and_illegal_moves_raise(self):
        tracker = PathTracker(chess.STARTING_FEN)
        with self.assertRaises(ValueError):
            tracker.back()
        with self.assertRaises(ValueError):
            tracker.push("e2e5")
        with self.assertRaises(ValueError):
            tracker.push("<claim_draw>")
        self.assertEqual(tracker.path, ())

    def test_repetition_matches_python_chess_along_the_path(self):
        tracker = PathTracker(chess.STARTING_FEN)
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
        tracker = PathTracker(chess.STARTING_FEN, repetition=4)
        self.assertEqual(tracker.repetition(), 4)
        for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
            tracker.push(move)
        self.assertEqual(tracker.repetition(), 5)
        tracker.push("e2e4")
        self.assertEqual(tracker.repetition(), 1)

    def test_board_tokens_are_lossless_against_python_chess(self):
        fens = [chess.STARTING_FEN, CASTLING_FEN]
        for fen in fens:
            with self.subTest(fen=fen):
                tracker = PathTracker(fen, repetition=2)
                decoded, repetition = decode_board(tracker.board_tokens())
                self.assertEqual(
                    decoded.split()[:5],
                    chess.Board(fen).fen(en_passant="fen").split()[:5],
                )
                self.assertEqual(repetition, 2)

    def test_deep_observation_keeps_en_passant_and_halfmove(self):
        tracker = PathTracker(chess.STARTING_FEN)
        tracker.push("e2e4")
        board = chess.Board()
        board.push_uci("e2e4")
        decoded, repetition = decode_board(tracker.board_tokens())
        self.assertEqual(decoded.split()[:5], board.fen(en_passant="fen").split()[:5])
        self.assertEqual(decoded.split()[3], "e3")
        self.assertEqual(decoded.split()[4], "0")
        self.assertEqual(repetition, 1)


class SearchTerminalTests(unittest.TestCase):
    """The search rule, cross-checked with python-chess's rule predicates."""

    def test_root_second_occurrence_is_not_terminal(self):
        tracker = PathTracker(chess.STARTING_FEN)
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
        tracker = PathTracker(chess.STARTING_FEN)
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

        # In check with an escape is still a 50-move draw.
        fen = "4k3/4R3/8/8/8/8/8/4K3 b - - 100 1"
        self.assertTrue(chess.Board(fen).is_check())
        self.assertFalse(chess.Board(fen).is_checkmate())
        self.assertEqual(PathTracker(fen).search_terminal(), "draw")

    def test_stalemate_and_insufficient_material(self):
        stalemate = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
        self.assertTrue(chess.Board(stalemate).is_stalemate())
        self.assertEqual(PathTracker(stalemate).search_terminal(), "draw")

        bare_kings = "8/8/8/4k3/8/8/8/4K3 w - - 0 1"
        self.assertTrue(chess.Board(bare_kings).is_game_over())
        self.assertFalse(chess.Board(bare_kings).is_stalemate())
        self.assertIsNone(PathTracker(bare_kings).search_terminal())

    def test_root_history_is_not_part_of_the_search_rule(self):
        tracker = PathTracker(chess.STARTING_FEN, repetition=5)
        self.assertEqual(tracker.repetition(), 5)
        self.assertIsNone(tracker.search_terminal())


if __name__ == "__main__":
    unittest.main()
