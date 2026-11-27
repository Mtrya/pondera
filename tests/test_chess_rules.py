"""Standard chess rules with frozen positions and move counts."""

import unittest

from pondera.chess import _rules as rules


def leaves(board, depth):
    if depth == 1:
        return len(board.legal_moves)
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += leaves(board, depth - 1)
        board.pop()
    return total


class RulesTests(unittest.TestCase):
    def test_standard_perft_counts(self):
        # Standard start and Kiwipete depth-three counts cover castling and pins.
        for fen, depth_three in (
            (rules.STARTING_FEN, 8_902),
            (
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                97_862,
            ),
        ):
            with self.subTest(fen=fen):
                board = rules.Board(fen)
                self.assertEqual(leaves(board, 3), depth_three)
                self.assertEqual(board.fen(en_passant="fen"), fen)
                self.assertEqual(board.move_stack, [])

    def test_fen_retains_raw_en_passant_and_repetition_uses_legal_ep(self):
        # The e-file rook pins the white pawn. It cannot capture on d6.
        fen = "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 19"
        board = rules.Board(fen)
        canonical = "k3r3/8/8/3pP3/8/8/8/4K3 w - - 0 19"
        self.assertEqual(board.fen(en_passant="fen"), fen)
        self.assertEqual(board.fen(), canonical)
        self.assertEqual(board.position_key(), rules.Board(canonical).position_key())
        with self.assertRaises(ValueError):
            board.push_uci("e5d6")
        self.assertEqual(board.fen(en_passant="fen"), fen)

    def test_san_only_disambiguates_legal_competitors(self):
        board = rules.Board("k3r3/8/8/8/8/4N1N1/8/4K3 w - - 0 1")
        self.assertEqual(board.parse_san("Nf5").uci(), "g3f5")
        self.assertEqual(board.san(rules.Move.from_uci("g3f5")), "Nf5")
        board = rules.Board("k7/8/8/8/8/4N1N1/8/4K3 w - - 0 1")
        with self.assertRaises(ValueError):
            board.parse_san("Nf5")
        self.assertEqual(board.parse_san("Ngf5").uci(), "g3f5")
        self.assertEqual(board.san(rules.Move.from_uci("g3f5")), "Ngf5")

    def test_game_over_uses_ordinary_draw_rules(self):
        board = rules.Board("6k1/8/8/8/8/8/8/K6R w - - 149 1")
        self.assertFalse(board.is_game_over())
        board.push_uci("h1h2")
        self.assertTrue(board.is_game_over())
        self.assertTrue(rules.Board("7k/8/8/8/8/8/8/KN6 w - - 0 1").is_game_over())
        self.assertFalse(rules.Board("7k/8/8/8/8/8/8/KNN5 w - - 0 1").is_game_over())

        board = rules.Board()
        for _ in range(3):
            for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
                board.push_uci(move)
        self.assertFalse(board.is_game_over())
        for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
            board.push_uci(move)
        self.assertTrue(board.is_game_over())

    def test_pgn_mainline_preserves_order_and_repetition_across_annotations(self):
        reason, rows, terminal = rules.replay_mainline(
            ';; preamble\n{ leading comment }\n[Event "Knight cycle"]\n\n'
            "1.Nf3 { comment\n\n continues } (1. e4 (1. d4 d5) e5) Nf6 $1\n"
            "2. Ng1 ; ignored variations and moves\n"
            "2... Ng8 *"
        )
        self.assertIsNone(reason)
        self.assertFalse(terminal)
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            rows[0], ("rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq -", 1, 1, 1)
        )
        self.assertEqual(rows[-1], (rules.STARTING_FEN.rsplit(" ", 2)[0], 2, 4, 3))
