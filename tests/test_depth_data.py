import json
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
from teacher_release import depth_row, play, write_depth_release

from pondera.chess.tokens import (
    MATE_LOSS,
    MATE_WIN,
    VALUE,
    decode_board,
)
from pondera.chess.mapping import UCI_MOVE_TO_IDX
from pondera.data.positions import DepthPositions
from pondera.data.teacher import prepare_teacher_depth

RUY_LOPEZ = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]
EXCHANGE_START = "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"
EXCHANGE_MOVES = ["b1c3", "g8f6"]

# Hand-derived from the frozen VALUE_BIN_BOUNDARIES contract: the first boundary
# index whose value is >= the teacher scalar. 0.5 sits in (0.2795, 0.6245] -> 15;
# the -1 and +1 atoms are the dedicated boundary-free bins 0 and 19.
BIN_OF_0_5 = 15
BIN_OF_MINUS_1 = 0
BIN_OF_PLUS_1 = 19


class DepthEncodingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        train_line = play(RUY_LOPEZ)
        val_line = play(EXCHANGE_MOVES, start=EXCHANGE_START)
        self.train_rows = []
        for fen, move, _ in train_line[:4]:
            self.train_rows.append(
                depth_row(fen, [(1, move, None, 0.5), (2, move, 3, 1.0)])
            )
        # A row missing depth 2 and a terminal row must be excluded.
        fen, move, _ = train_line[4]
        self.train_rows.append(depth_row(fen, [(1, move, None, 0.0)]))
        terminal = chess.Board()
        for move in ["f2f3", "e7e5", "g2g4", "d8h4"]:
            terminal.push_uci(move)
        self.train_rows.append({"fen": terminal.fen(), "terminal": True})
        self.val_rows = [
            depth_row(fen, [(1, move, -2, -1.0), (2, move, None, -1.0)])
            for fen, move, _ in val_line
        ]
        self.data = self.root / "encoded"

    def encode(self, max_depth=2):
        release = write_depth_release(
            self.root, {"train": self.train_rows, "val": self.val_rows}
        )
        return prepare_teacher_depth(release, self.data, max_depth=max_depth)

    def test_manifest_rows_and_exclusions(self):
        metadata = self.encode()
        self.assertEqual(metadata["kind"], "teacher-depth")
        self.assertEqual(metadata["depths"], [1, 2])
        self.assertEqual(metadata["rows_train"], 4)
        self.assertEqual(metadata["rows_val"], 2)
        self.assertEqual(
            metadata["excluded"]["train"], {"incomplete_depths": 1, "terminal": 1}
        )
        self.assertEqual(metadata["excluded"]["val"], {})

    def test_targets_follow_the_frozen_vocabulary_contract(self):
        self.encode()
        dataset = DepthPositions(self.data, "train", legal=True)
        self.assertEqual(len(dataset), 4)
        for index, row in enumerate(self.train_rows[:4]):
            fen = row["fen"]
            move = row["unit"]["per_depth"][0]["r1"]["bm"]
            self.assertEqual(decode_board(dataset.tokens[index]), (fen, 1))
            self.assertEqual(int(dataset.moves[index, 0]), UCI_MOVE_TO_IDX[move])
            # d1 value 0.5 -> bin 15; d2 mate +3 -> MATE_WIN_3.
            self.assertEqual(int(dataset.value_tokens[index, 0]), VALUE + BIN_OF_0_5)
            self.assertEqual(int(dataset.value_tokens[index, 1]), MATE_WIN + 3)
            self.assertAlmostEqual(float(dataset.values[index, 1]), 1.0)
            legal = dataset.legal[index]
            legal = set(legal[legal >= 0].tolist())
            # Independent source: python-chess legal move generation.
            self.assertEqual(
                legal,
                {UCI_MOVE_TO_IDX[m.uci()] for m in chess.Board(fen).legal_moves},
            )
        val = DepthPositions(self.data, "val")
        # val d1 mate -2 -> MATE_LOSS_2; d2 value -1 -> the dedicated bin 0.
        self.assertEqual(int(val.value_tokens[0, 0]), MATE_LOSS + 2)
        self.assertEqual(int(val.value_tokens[0, 1]), VALUE + BIN_OF_MINUS_1)
        self.assertEqual(val.value_tokens.dtype, np.int16)
        self.assertEqual(val.moves.shape, (2, 2))
        self.assertEqual(val.values.shape, (2, 2))

    def test_value_atoms_and_dataset_guardrails(self):
        # Rebuild with the +1 atom replacing the first validation row's targets.
        self.val_rows[0] = depth_row(
            self.val_rows[0]["fen"],
            [(1, "b1c3", None, 1.0), (2, "b1c3", None, 1.0)],
        )
        self.encode()
        val = DepthPositions(self.data, "val")
        self.assertEqual(int(val.value_tokens[0, 0]), VALUE + BIN_OF_PLUS_1)
        manifest_path = self.data / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        del manifest["kind"]
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            DepthPositions(self.data, "val")


if __name__ == "__main__":
    unittest.main()
