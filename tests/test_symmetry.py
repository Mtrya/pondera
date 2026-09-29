import unittest

import chess
import numpy as np
import torch

from pondera.chess.board_tokens import decode_board, encode_boards
from pondera.chess.mapping import IDX_TO_UCI_MOVE, UCI_MOVE_TO_IDX
from pondera.chess.symmetry import (
    NUM_TRANSFORMS,
    apply_to_legal,
    apply_to_moves,
    apply_to_tokens,
)
from pondera.training.supervised import PolicyCollator

FENS = [
    chess.STARTING_FEN,
    "r3k2r/8/8/8/3pP3/8/8/R3K2R b Kq e3 130 400",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq h6 0 12",
    "8/2P5/8/8/4k3/8/6p1/4K3 w - - 0 1",
    "4k3/8/8/2pP4/8/8/8/4K3 b - c6 5 30",
    "7k/8/8/8/8/8/p7/7K b - - 0 80",
]


class SymmetryTokenTests(unittest.TestCase):
    def test_token_transforms_match_python_chess(self):
        transforms = [0, 1, 0, 1, 1, 0]
        ids = encode_boards(FENS, [1, 2, 3, 1, 2, 3])
        changed = apply_to_tokens(ids, transforms)
        self.assertEqual(changed.dtype, ids.dtype)
        for index, fen in enumerate(FENS):
            transformed, repetition = decode_board(changed[index])
            expected = chess.Board(fen)
            if transforms[index]:
                expected = expected.mirror()
            self.assertEqual(transformed, expected.fen(en_passant="fen"))
            self.assertEqual(repetition, [1, 2, 3, 1, 2, 3][index])

    def test_mirror_is_an_involution(self):
        ids = encode_boards(FENS)
        draws = np.ones(len(FENS), dtype=np.int64)
        twice = apply_to_tokens(apply_to_tokens(ids, draws), draws)
        np.testing.assert_array_equal(twice, ids)


class SymmetryMoveTests(unittest.TestCase):
    def test_move_permutation_matches_mirrored_legal_moves(self):
        for fen in FENS:
            board = chess.Board(fen)
            original = {m.uci() for m in board.legal_moves}
            expected = {m.uci() for m in board.mirror().legal_moves}
            moves = np.asarray([UCI_MOVE_TO_IDX[u] for u in original])
            draws = np.ones(len(moves), dtype=np.int64)
            actual = {IDX_TO_UCI_MOVE[i] for i in apply_to_moves(moves, draws)}
            self.assertEqual(actual, expected)

    def test_move_permutation_is_an_involution(self):
        size = len(IDX_TO_UCI_MOVE)
        moves = np.arange(size)
        draws = np.ones(size, dtype=np.int64)
        twice = apply_to_moves(apply_to_moves(moves, draws), draws)
        np.testing.assert_array_equal(twice, moves)

    def test_legal_arrays_keep_their_fill_value(self):
        legal = np.asarray([[0, 5, -1, -1], [7, 9, 11, -1]])
        draws = np.ones(2, dtype=np.int64)
        changed = apply_to_legal(legal, draws)
        np.testing.assert_array_equal(changed[legal < 0], -1)
        self.assertTrue((changed[legal >= 0] >= 0).all())
        np.testing.assert_array_equal(apply_to_legal(changed, draws), legal)


class SymmetryCollatorTests(unittest.TestCase):
    def rows(self):
        legal = []
        for fen in FENS:
            board = chess.Board(fen)
            moves = [UCI_MOVE_TO_IDX[m.uci()] for m in board.legal_moves]
            legal.append(moves + [-1] * (98 - len(moves)))
        ids = encode_boards(FENS)
        targets = np.asarray([row[0] for row in legal])
        return [
            (ids[i], targets[i], np.asarray(legal[i], dtype=np.int64))
            for i in range(len(FENS))
        ]

    def test_mirror_mode_is_deterministic_given_torch_seed(self):
        rows = self.rows()
        collate = PolicyCollator("pieces", "mirror")
        torch.manual_seed(640)
        first_inputs, first_targets, first_meta = collate(rows)
        torch.manual_seed(640)
        second_inputs, second_targets, second_meta = collate(rows)
        torch.testing.assert_close(first_inputs["ids"], second_inputs["ids"])
        torch.testing.assert_close(first_targets, second_targets)
        torch.testing.assert_close(first_meta["legal"], second_meta["legal"])

    def test_none_mode_leaves_everything_untouched(self):
        rows = self.rows()
        plain_inputs, plain_targets, plain_meta = PolicyCollator("pieces")(rows)
        none_inputs, none_targets, none_meta = PolicyCollator("pieces", "none")(rows)
        torch.testing.assert_close(plain_inputs["ids"], none_inputs["ids"])
        torch.testing.assert_close(plain_targets, none_targets)
        torch.testing.assert_close(plain_meta["legal"], none_meta["legal"])


if __name__ == "__main__":
    unittest.main()
