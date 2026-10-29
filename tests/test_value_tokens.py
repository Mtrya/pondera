import unittest

from pondera.chess.codec import (
    VALUE_BIN_BOUNDARIES,
    VALUE_BIN_REPRESENTATIVES,
    decode_value_token,
    mate_token,
    value_bin,
)
from pondera.chess.vocabulary import (
    BOARD_TOKENS,
    CASTLING,
    CHESS_VOCAB_SIZE,
    DELIM,
    EN_PASSANT,
    END,
    FINAL,
    HALFMOVE,
    LINE,
    MATE_LOSS,
    MATE_WIN,
    META_END,
    MOVE,
    PAD,
    REPETITION,
    SIDE,
    VALUE,
)


class ValueBinTests(unittest.TestCase):
    """Frozen contract: 20 bins, atom bins for +-1, (lo, hi] membership."""

    def test_atoms_are_dedicated_bins(self):
        self.assertEqual(value_bin(-1.0), 0)
        self.assertEqual(value_bin(1.0), 19)
        # The nearest 0.001-grid values fall into the adjacent interior bins.
        self.assertEqual(value_bin(-0.999), 1)
        self.assertEqual(value_bin(0.999), 18)

    def test_boundaries_belong_to_their_upper_bin(self):
        # Spot checks on the 0.001 teacher grid against the frozen boundaries.
        for value, expected in [
            (-0.9935, 1),
            (-0.993, 2),
            (-0.0685, 6),
            (-0.068, 7),
            (-0.0005, 9),
            (0.0, 10),
            (0.0045, 10),
            (0.005, 11),
            (0.9995, 18),
        ]:
            with self.subTest(value=value):
                self.assertEqual(value_bin(value), expected)

    def test_bin_structure_and_representatives(self):
        self.assertEqual(len(VALUE_BIN_BOUNDARIES), 19)
        self.assertEqual(len(VALUE_BIN_REPRESENTATIVES), 20)
        representatives = list(VALUE_BIN_REPRESENTATIVES)
        self.assertEqual(representatives, sorted(representatives))
        self.assertEqual(representatives[0], -1.0)
        self.assertEqual(representatives[-1], 1.0)
        # Interior representatives lie inside their (lo, hi] interval.
        for index in range(1, 19):
            lo, hi = VALUE_BIN_BOUNDARIES[index - 1], VALUE_BIN_BOUNDARIES[index]
            self.assertLess(lo, representatives[index])
            self.assertLessEqual(representatives[index], hi)

    def test_out_of_range_values_are_rejected(self):
        for bad in (-1.001, 1.001, float("nan")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    value_bin(bad)


class UnifiedVocabularyTests(unittest.TestCase):
    """Frozen id layout: board and metadata atoms, moves, values, delimiters."""

    def test_frozen_offsets(self):
        # Hand arithmetic over the frozen segment sizes: 768 board atoms,
        # 181 metadata atoms (2 side + 8 castling + 65 ep + 101 halfmove +
        # 5 repetition), then the 1,969-move block and the frozen value,
        # mate, and delimiter blocks.
        self.assertEqual(BOARD_TOKENS, 768)
        self.assertEqual((SIDE, CASTLING, EN_PASSANT), (768, 770, 778))
        self.assertEqual((HALFMOVE, REPETITION, META_END), (843, 944, 948))
        self.assertEqual(MOVE, 949)
        self.assertEqual(MOVE, META_END + 1)
        self.assertEqual(VALUE, 949 + 1969)
        self.assertEqual(MATE_WIN, VALUE + 20)
        self.assertEqual(MATE_LOSS, MATE_WIN + 29)
        self.assertEqual(DELIM, MATE_LOSS + 31)
        self.assertEqual((PAD, END, LINE, FINAL), (2998, 2999, 3000, 3001))
        self.assertEqual(CHESS_VOCAB_SIZE, 3002)

    def test_value_decoder_rejects_non_value_tokens(self):
        # The metadata atoms end at 948; the move block starts the unified
        # sequence space and the delimiters close it.
        self.assertEqual(CHESS_VOCAB_SIZE, DELIM + 4)
        for token in (MOVE, META_END, PAD, END, LINE, FINAL):
            with self.assertRaises(ValueError):
                decode_value_token(token)

    def test_mate_tokens_cover_the_frozen_distances(self):
        self.assertEqual(mate_token(0, win=True), MATE_WIN)
        self.assertEqual(mate_token(28, win=True), MATE_WIN + 28)
        self.assertEqual(mate_token(0, win=False), MATE_LOSS)
        self.assertEqual(mate_token(30, win=False), MATE_LOSS + 30)
        for distance, win in ((-1, True), (29, True), (-1, False), (31, False)):
            with self.subTest(distance=distance, win=win):
                with self.assertRaises(ValueError):
                    mate_token(distance, win)

    def test_decode_value_token(self):
        self.assertEqual(decode_value_token(VALUE), -1.0)
        self.assertEqual(decode_value_token(VALUE + 19), 1.0)
        for distance in (0, 1, 28):
            self.assertEqual(decode_value_token(mate_token(distance, True)), 1.0)
        for distance in (0, 1, 30):
            self.assertEqual(decode_value_token(mate_token(distance, False)), -1.0)
        with self.assertRaises(ValueError):
            decode_value_token(MOVE)  # a move token carries no value decode
        with self.assertRaises(ValueError):
            decode_value_token(CHESS_VOCAB_SIZE)


if __name__ == "__main__":
    unittest.main()
