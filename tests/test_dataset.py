import json
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch
from dataset_fixtures import encoded_row, write_trace_dataset

from pondera.chess.codec import encode_board
from pondera.chess.vocabulary import (
    CHESS_TOKENIZER_VERSION,
    END,
    FINAL,
    MOVE,
    PAD,
    UCI_MOVE_TO_IDX,
)
from pondera.data.dataset import DATA_FORMAT_VERSION, TracePositions, write_trace_rows
from pondera.data.trace import (
    ROLE_CONTROL,
    ROLE_FINAL_EVAL,
    ROLE_FINAL_MOVE,
    ROLE_PAD,
    TRACE_GRAMMAR_VERSION,
    TraceWriter,
)
from pondera.models.policy import create_model
from pondera.training.trace import TraceCollator

# A black-to-move root reached after 1.e4, and a castling position with both
# sides holding all rights.
BLACK_TO_MOVE = chess.Board()
BLACK_TO_MOVE.push_uci("e2e4")
BLACK_TO_MOVE = BLACK_TO_MOVE.fen(en_passant="fen")
CASTLING_FEN = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"


def move_token(uci):
    return MOVE + UCI_MOVE_TO_IDX[uci]


def counter_row(fen, repetition=1):
    """A small v2 trace: a candidate, its evaluation, and the label suffix."""
    board = chess.Board(fen)
    legal = sorted(move.uci() for move in board.legal_moves)
    writer = TraceWriter(fen, repetition)
    writer.depth()
    writer.move(legal[0])
    writer.evaluate(value=0.25)
    writer.back()
    writer.evaluate(value=0.25)
    writer.final(legal[1])
    writer.final_evaluate(value=0.25)
    after = chess.Board(fen)
    after.push_uci(legal[1])
    writer.final_pv(sorted(move.uci() for move in after.legal_moves)[0])
    writer.end()
    return encoded_row(fen, writer.tokens, writer.roles, repetition)


class DatasetContractTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        fens = [chess.STARTING_FEN, BLACK_TO_MOVE, CASTLING_FEN]
        self.rows = [counter_row(fen) for fen in fens]
        self.data = write_trace_dataset(
            self.root / "data", {"train": self.rows, "val": self.rows[:2]}
        )

    def test_manifest_carries_the_current_grammar(self):
        manifest = json.loads((self.data / "release.json").read_text())
        self.assertEqual(manifest["grammar_version"], TRACE_GRAMMAR_VERSION)
        self.assertEqual(manifest["grammar_version"], 2)
        self.assertEqual(manifest["format_version"], DATA_FORMAT_VERSION)
        self.assertEqual(manifest["format_version"], 8)
        self.assertEqual(manifest["tokenizer_version"], CHESS_TOKENIZER_VERSION)

    def test_rejects_old_trace_formats(self):
        manifest_path = self.data / "release.json"
        manifest = json.loads(manifest_path.read_text())
        for version in (5, 6, 7):
            with self.subTest(version=version):
                manifest["format_version"] = version
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    TracePositions(self.data, "train")

    def test_dataset_rejects_a_missing_grammar_version(self):
        manifest_path = self.data / "release.json"
        manifest = json.loads(manifest_path.read_text())
        del manifest["grammar_version"]
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            TracePositions(self.data, "train")

    def test_dataset_arrays_and_roles(self):
        positions = TracePositions(self.data, "train")
        self.assertEqual(len(positions), 3)
        boards, trace, length, roles = positions[0]
        self.assertEqual(len(trace), length)
        self.assertEqual(int(length), len(self.rows[0]["trace"]))
        self.assertEqual(trace[:length].tolist(), self.rows[0]["trace"].tolist())
        self.assertEqual(roles[:length].tolist(), self.rows[0]["roles"].tolist())
        self.assertTrue(np.all(roles[length:] == ROLE_PAD))
        # The decision follows FINAL, and the trace closes with <end>.
        self.assertEqual(int(trace[length - 1]), END)
        decision = int(np.flatnonzero(roles[:length] == ROLE_FINAL_MOVE)[0])
        self.assertEqual(int(trace[decision - 1]), FINAL)
        self.assertNotEqual(decision, length - 1)
        self.assertEqual(int(roles[length - 1]), ROLE_CONTROL)

    def test_ragged_rows_cross_parquet_row_groups(self):
        rows = [self.rows[index % len(self.rows)] for index in range(130)]
        directory = write_trace_dataset(self.root / "groups", {"train": rows})
        positions = TracePositions(directory, "train")
        for index in (129, 0, 63, 64, 127, 128, 1):
            board, tokens, length, roles = positions[index]
            expected = rows[index]
            self.assertEqual(tokens.tolist(), expected["trace"].tolist())
            self.assertEqual(roles.tolist(), expected["roles"].tolist())
            self.assertEqual(length, len(tokens))
            self.assertEqual(board.tolist(), encode_board(expected["fen"]).tolist())

    def test_collator_pads_tokens_and_locates_the_final_decision(self):
        positions = TracePositions(self.data, "train")
        rows = [positions[index] for index in range(len(positions))]
        batch = TraceCollator()(rows)
        width = max(int(row[2]) for row in rows)
        self.assertEqual(batch["trace_ids"].shape, (len(rows), width))
        self.assertEqual(batch["trace_roles"].shape, (len(rows), width))
        self.assertEqual(batch["trace_ids"].dtype, torch.int16)
        self.assertEqual(batch["trace_roles"].dtype, torch.int8)
        for index, (_, trace, length, roles) in enumerate(rows):
            self.assertEqual(
                batch["trace_ids"][index, :length].tolist(), trace[:length].tolist()
            )
            self.assertEqual(
                batch["trace_roles"][index, :length].tolist(), roles[:length].tolist()
            )
            self.assertTrue(bool((batch["trace_ids"][index, length:] == PAD).all()))
            self.assertTrue(
                bool((batch["trace_roles"][index, length:] == ROLE_PAD).all())
            )
            decision = int(batch["final_move_index"][index])
            self.assertEqual(int(roles[decision]), ROLE_FINAL_MOVE)
            self.assertEqual(int(trace[decision - 1]), FINAL)
            self.assertEqual(int(roles[length - 1]), ROLE_CONTROL)


class CollatorContractTests(unittest.TestCase):
    def setUp(self):
        self.writer = TraceWriter(chess.STARTING_FEN)
        self.writer.depth()
        self.writer.move("e2e4")
        self.writer.evaluate(value=0.25)
        self.writer.back()
        self.writer.final("d2d4")
        self.writer.final_evaluate(value=0.25)
        self.writer.final_pv("e7e5")
        self.writer.end()
        self.trace = self.writer.tokens.copy()
        self.roles = self.writer.roles.copy()

    def test_final_move_index_is_the_decision_not_the_end(self):
        batch = TraceCollator()(
            [
                (
                    encode_board(chess.STARTING_FEN),
                    self.trace,
                    len(self.trace),
                    self.roles,
                )
            ]
        )
        decision = int(batch["final_move_index"][0])
        self.assertEqual(int(self.trace[decision]), move_token("d2d4"))
        self.assertNotEqual(decision, len(self.trace) - 1)
        self.assertEqual(int(self.roles[decision + 1]), ROLE_FINAL_EVAL)


def write_shard(directory, index, fen):
    """Publish one one-row shard whose board array is exactly its own width."""
    board = chess.Board(fen)
    writer = TraceWriter(fen)
    writer.final(sorted(move.uci() for move in board.legal_moves)[0])
    writer.final_evaluate(value=0.0)
    writer.end()
    shard = directory / f"shard-{index:06d}"
    shard.mkdir()
    write_trace_rows(
        shard,
        [
            {
                "split": "train",
                "board_tokens": encode_board(fen),
                "trace_tokens": writer.tokens,
                "trace_len": len(writer.tokens),
                "trace_roles": writer.roles,
                **{
                    name: []
                    for name in (
                        "state_without_board",
                        "state_with_board",
                        "state_best",
                        "state_basis",
                        "state_value",
                    )
                },
            }
        ],
    )
    return shard.name


class MixedShardCollationTests(unittest.TestCase):
    def test_board_rows_are_padded_to_the_batch_maximum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                write_shard(root, 0, CASTLING_FEN),
                write_shard(root, 1, chess.STARTING_FEN),
            ]
            (root / "release.json").write_text(
                json.dumps(
                    {
                        "format_version": DATA_FORMAT_VERSION,
                        "grammar_version": TRACE_GRAMMAR_VERSION,
                        "tokenizer_version": CHESS_TOKENIZER_VERSION,
                        "shards": [{"path": path, "rows": 1} for path in paths],
                    }
                )
            )
            positions = TracePositions(root, "train")
            rows = [positions[0], positions[1]]
            widths = [len(row[0]) for row in rows]
            self.assertLess(widths[0], widths[1])

            batch = TraceCollator()(rows)
            self.assertEqual(batch["ids"].shape, (len(rows), max(widths)))
            for index, (board, trace, length, roles) in enumerate(rows):
                self.assertEqual(
                    batch["ids"][index, : widths[index]].tolist(), board.tolist()
                )
                self.assertTrue(
                    bool((batch["ids"][index, widths[index] :] == PAD).all())
                )
                self.assertEqual(
                    batch["trace_ids"][index, :length].tolist(),
                    trace[:length].tolist(),
                )

            # Padding is masked, so a mixed batch predicts each row exactly as
            # that row does when collated alone at its own width.
            model = create_model(
                {
                    "num_blocks": 1,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_heads": 2,
                }
            ).eval()
            with torch.no_grad():
                batched = model.trace_logits(batch["ids"], batch["trace_ids"])
                for index, row in enumerate(rows):
                    alone_batch = TraceCollator()([row])
                    alone = model.trace_logits(
                        alone_batch["ids"], alone_batch["trace_ids"]
                    )
                    torch.testing.assert_close(batched[index], alone[0])


if __name__ == "__main__":
    unittest.main()
