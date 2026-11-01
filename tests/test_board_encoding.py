import unittest

import chess
import numpy as np
import torch

from pondera.chess.codec import decode_board, encode_board, encode_boards
from pondera.chess.domain import PIECE_TO_IDX
from pondera.chess.tracker import repetition_count
from pondera.chess.vocabulary import SIDE
from pondera.models.board import black_to_move, reflect_board_tokens


def assert_decoded_board(test, tokens, fen, repetition):
    """Check the FEN modulo the fullmove counter and saturated counters."""
    decoded, decoded_repetition = decode_board(tokens)
    test.assertEqual(decoded.split()[:4], fen.split()[:4])
    test.assertEqual(decoded.split()[4], str(min(int(fen.split()[4]), 100)))
    test.assertEqual(decoded.split()[5], "1")
    test.assertEqual(decoded_repetition, min(repetition, 5))


class BoardEncodingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_board_fields_and_counter_atoms(self):
        fens = [
            chess.STARTING_FEN,
            "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 99 257",
            "7k/Q7/8/8/8/8/8/K7 b - - 100 400",
            "7k/Q7/8/8/8/8/8/K7 b - - 137 500",
        ]
        repetitions = [1, 4, 9, 2]
        batch = encode_boards(fens, repetitions)
        for fen, rep, tokens in zip(fens, repetitions, batch):
            assert_decoded_board(self, tokens, fen, rep)
        for bad in [
            "8/8/8/8/8/8/8/9 w - - 0 1",
            "8/8/8/8/8/8/8/8 x - - 0 1",
            "8/8/8/8/8/8/8/8 w - - -1 1",
        ]:
            with self.assertRaises(ValueError):
                encode_board(bad)
        with self.assertRaises(ValueError):
            encode_board(chess.STARTING_FEN, 0)

    def test_sparse_atoms_match_python_chess_and_reject_duplicates(self):
        fen = chess.STARTING_FEN
        tokens = encode_board(fen, 1)
        self.assertEqual(len(tokens), 32 + 8)
        expected = sorted(
            square * 12 + PIECE_TO_IDX[piece.symbol()]
            for square, piece in chess.Board(fen).piece_map().items()
        )
        self.assertEqual(tokens[:32].tolist(), expected)
        self.assertEqual(tokens[32], SIDE)
        duplicated = np.concatenate((tokens[:2], tokens[:1], tokens[1:]))
        with self.assertRaises(ValueError):
            decode_board(duplicated)

    def test_repetition_count_matches_played_occurrences(self):
        board = chess.Board()
        cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
        # Three four-ply knight cycles return to the start: four occurrences.
        for _ in range(3):
            for move in cycle:
                board.push_uci(move)
        self.assertEqual(repetition_count(board), 4)
        # A fourth cycle makes five, the shared top repetition atom.
        for move in cycle:
            board.push_uci(move)
        self.assertEqual(repetition_count(board), 5)
        for move in cycle:
            board.push_uci(move)
        self.assertEqual(repetition_count(board), 5)

    def test_orientation_is_reversible_and_matches_chess_mirror(self):
        fens = [
            "r3k2r/8/8/8/3pP3/8/8/R3K2R b Kq e3 99 400",
            "7k/8/8/8/8/8/p7/7K b - - 0 80",
            chess.STARTING_FEN,
        ]
        ids = torch.from_numpy(encode_boards(fens, [3, 2, 1]))
        black = black_to_move(ids)
        changed = reflect_board_tokens(ids, black)
        torch.testing.assert_close(reflect_board_tokens(changed, black), ids)
        for index, fen in enumerate(fens):
            expected = chess.Board(fen)
            if expected.turn == chess.BLACK:
                expected = expected.mirror()
            parts = expected.fen(en_passant="fen").split()
            # The side atom and the encoded counters keep the root frame.
            parts[1] = fen.split()[1]
            parts[5] = "1"
            assert_decoded_board(
                self, changed[index], " ".join(parts), [3, 2, 1][index]
            )


if __name__ == "__main__":
    unittest.main()
