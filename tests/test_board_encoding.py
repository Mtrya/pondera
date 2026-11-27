import unittest

import numpy as np
import torch

from pondera.chess import _rules as rules
from pondera.chess.codec import decode_board, encode_board, encode_boards
from pondera.chess.symmetry import TOKEN_MIRROR
from pondera.chess.tracker import repetition_count
from pondera.chess.vocabulary import SIDE


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
            rules.STARTING_FEN,
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
            encode_board(rules.STARTING_FEN, 0)

    def test_starting_position_atom_ids_and_duplicate_rejection(self):
        fen = rules.STARTING_FEN
        tokens = encode_board(fen, 1)
        self.assertEqual(len(tokens), 32 + 8)
        # Arithmetic vocabulary IDs for the occupied starting squares a1–h8.
        expected = [
            3,
            13,
            26,
            40,
            53,
            62,
            73,
            87,
            96,
            108,
            120,
            132,
            144,
            156,
            168,
            180,
            582,
            594,
            606,
            618,
            630,
            642,
            654,
            666,
            681,
            691,
            704,
            718,
            731,
            740,
            751,
            765,
        ]
        self.assertEqual(tokens[:32].tolist(), expected)
        self.assertEqual(tokens[32], SIDE)
        duplicated = np.concatenate((tokens[:2], tokens[:1], tokens[1:]))
        with self.assertRaises(ValueError):
            decode_board(duplicated)

    def test_repetition_count_matches_played_occurrences(self):
        board = rules.Board()
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
            rules.STARTING_FEN,
        ]
        ids = torch.from_numpy(encode_boards(fens, [3, 2, 1])).long()
        black = torch.tensor([True, True, False])
        mirror = torch.from_numpy(TOKEN_MIRROR)
        changed = torch.where(black[:, None], mirror[ids], ids)
        restored = torch.where(black[:, None], mirror[changed], changed)
        torch.testing.assert_close(restored, ids)
        expected = [
            "r3k2r/8/8/3Pp3/8/8/8/R3K2R b Qk e6 99 400",
            "7k/P7/8/8/8/8/8/7K b - - 0 80",
            rules.STARTING_FEN,
        ]
        for tokens, fen, repetition in zip(changed, expected, [3, 2, 1], strict=True):
            assert_decoded_board(self, tokens, fen, repetition)


if __name__ == "__main__":
    unittest.main()
