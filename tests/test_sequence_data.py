import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch
from teacher_release import (
    candidate_entries,
    candidate_row,
    play,
    write_sequence_dataset,
)

from pondera.chess.mapping import IDX_TO_UCI_MOVE, UCI_MOVE_TO_IDX
from pondera.chess.symmetry import move_permutations, reflect_uci
from pondera.chess.tokens import (
    END,
    FINAL,
    LINE,
    MATE_LOSS,
    MATE_WIN,
    MOVE,
    PAD,
    VALUE,
    encode_board,
)
from pondera.data.positions import SequencePositions
from pondera.data.trace import (
    LEVEL0,
    LEVEL15,
    PV_LIMIT,
    absolute_uci,
    depth_snapshot,
    encode_trace,
    is_eval_token,
    is_move_token,
    max_segments,
    move_token,
    parse_trace,
)
from pondera.training.sequence import SequenceCollator

RUY_LOPEZ = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]
# A black-to-move record of the same line and a middlegame record with few pieces.
BLACK_TO_MOVE = play(RUY_LOPEZ)[1][0]
EXCHANGE_START = "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"
FENS = [chess.STARTING_FEN, BLACK_TO_MOVE, EXCHANGE_START]

# Parsed from the frozen teacher contract: value 0.5 lies in bin 15, the +-1
# atoms are bins 19 and 0, and mate distances offset their block base.
BIN_OF_0_5 = 15


def replay_board(board, black):
    return board.mirror() if black else board.copy()


class TraceEncodingTests(unittest.TestCase):
    def test_level15_roundtrip_matches_teacher_rows(self):
        for fen in FENS:
            with self.subTest(fen=fen):
                board = chess.Board(fen)
                black = board.turn == chess.BLACK
                entries = candidate_entries(board)
                row = candidate_row(fen, entries)
                tokens, bounds = encode_trace(depth_snapshot(row), black, LEVEL15)
                segments, error = parse_trace(tokens)
                self.assertIsNone(error)
                self.assertEqual(len(bounds), len(segments))
                self.assertEqual(bounds[-1][1], len(tokens))
                # Bounds tile the trace in order and mark every token once.
                self.assertEqual(bounds[0][0], 0)
                for (_, end), (start, _) in zip(bounds, bounds[1:]):
                    self.assertEqual(end, start)
                for entry, segment in zip(entries, segments[:-1]):
                    self.assertEqual(segment.kind, "line")
                    moves = absolute_uci(segment.moves, black)
                    self.assertEqual(moves, entry[1][:PV_LIMIT])
                    self.assertTrue(all(is_move_token(t) for t in segment.moves))
                    self.assertTrue(is_eval_token(segment.evaluation))
                    # python-chess replays both frames: the stored canonical
                    # moves on the canonical board and the decoded absolute
                    # moves on the source board.
                    canonical = replay_board(board, black)
                    for token in segment.moves:
                        canonical.push_uci(IDX_TO_UCI_MOVE[token - MOVE])
                    original = board.copy()
                    for move in moves:
                        original.push_uci(move)
                final = segments[-1]
                self.assertEqual(final.kind, "final")
                self.assertTrue(final.terminated)
                self.assertEqual(absolute_uci(final.moves, black), [entries[0][0]])
                self.assertEqual(final.evaluation, segments[0].evaluation)

    def test_level0_is_the_final_segment_of_level15(self):
        for fen in FENS:
            with self.subTest(fen=fen):
                board = chess.Board(fen)
                black = board.turn == chess.BLACK
                row = candidate_row(fen, candidate_entries(board))
                snapshot = depth_snapshot(row)
                level0, bounds0 = encode_trace(snapshot, black, LEVEL0)
                level15, bounds15 = encode_trace(snapshot, black, LEVEL15)
                self.assertEqual(bounds0, [(0, len(level0))])
                self.assertEqual(len(level0), 4)
                start, end = bounds15[-1]
                self.assertEqual(level15[start:end], level0)
                segments, error = parse_trace(level0)
                self.assertIsNone(error)
                self.assertEqual([segment.kind for segment in segments], ["final"])

    def test_evaluations_follow_the_frozen_value_and_mate_contract(self):
        fen = FENS[0]
        board = chess.Board(fen)
        entries = candidate_entries(board, count=1)
        pv = entries[0][1]
        candidates = [
            (pv[0], pv, None, 0.5),
            (entries[0][0], pv, 3, 1.0),
            (entries[0][0], pv, -2, -1.0),
        ]
        row = candidate_row(fen, candidates)
        tokens, _ = encode_trace(depth_snapshot(row), False, LEVEL15)
        segments, error = parse_trace(tokens)
        self.assertIsNone(error)
        self.assertEqual(segments[0].evaluation, VALUE + BIN_OF_0_5)
        self.assertEqual(segments[1].evaluation, MATE_WIN + 3)
        self.assertEqual(segments[2].evaluation, MATE_LOSS + 2)

    def test_black_to_move_moves_are_rank_reflected(self):
        fen = BLACK_TO_MOVE
        board = chess.Board(fen)
        self.assertEqual(board.turn, chess.BLACK)
        entries = candidate_entries(board)
        row = candidate_row(fen, entries)
        tokens, _ = encode_trace(depth_snapshot(row), True, LEVEL15)
        segments, error = parse_trace(tokens)
        self.assertIsNone(error)
        expected = [reflect_uci(move) for move in entries[0][1][:PV_LIMIT]]
        stored = [IDX_TO_UCI_MOVE[token - MOVE] for token in segments[0].moves]
        self.assertEqual(stored, expected)
        self.assertEqual(
            absolute_uci(segments[0].moves, True), entries[0][1][:PV_LIMIT]
        )
        self.assertNotEqual(expected, entries[0][1][:PV_LIMIT])
        self.assertIn(chess.Move.from_uci(expected[0]), board.mirror().legal_moves)
        for move in expected:
            self.assertEqual(reflect_uci(reflect_uci(move)), move)
        replay = board.mirror()
        for move in expected:
            replay.push_uci(move)

    def test_encoder_reflection_matches_the_policy_action_table(self):
        # The encoding frame and the policy's action permutation must be the
        # same transform, since both move black-to-move data into one frame.
        for uci in ("e2e4", "a7a5", "e7e8q", "g1f3", "e1g1", "<claim_draw>"):
            expected = int(move_permutations()[1][UCI_MOVE_TO_IDX[uci]])
            self.assertEqual(move_token(uci, True) - MOVE, expected)
            self.assertEqual(move_token(uci, False), MOVE + UCI_MOVE_TO_IDX[uci])

    def test_parse_reports_broken_programs(self):
        move, evaluation = MOVE + 7, VALUE + 5
        cases = {
            "final_missing_end": [FINAL, move, evaluation],
            "final_missing_evaluation": [FINAL, move, END],
            "final_without_move": [FINAL, VALUE, END],
            "line_without_move": [LINE, evaluation],
            "line_missing_evaluation": [LINE, move, END],
            "truncated": [LINE, move],
            "missing_final": [LINE, move, evaluation],
            "trailing_tokens": [FINAL, move, evaluation, END, evaluation],
            "unexpected_token": [evaluation],
        }
        for expected, tokens in cases.items():
            with self.subTest(error=expected):
                _, error = parse_trace(tokens)
                self.assertEqual(error, expected)
        _, error = parse_trace([])
        self.assertEqual(error, "missing_final")
        segments, error = parse_trace([FINAL, move, evaluation, END, PAD, PAD])
        self.assertIsNone(error)
        self.assertEqual(len(segments), 1)


class SegmentOrderTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        rows = []
        for fen in FENS:
            board = chess.Board(fen)
            for plies in (2, 4, 6, 8):
                rows.append(candidate_row(fen, candidate_entries(board, plies=plies)))
        self.data = write_sequence_dataset(
            self.root, {"train": rows, "val": rows[:2]}, level=LEVEL15
        )

    def test_shuffled_presentation_preserves_content_and_puts_final_last(self):
        data = SequencePositions(self.data, "train")
        rows = [data[index] for index in range(len(data))]
        plain = SequenceCollator(shuffle=False)(rows)
        torch.manual_seed(0)
        shuffled = SequenceCollator(shuffle=True)(rows)
        self.assertEqual(plain["trace_ids"].shape, shuffled["trace_ids"].shape)
        changed = 0
        for index, (_, trace, trace_len, bounds) in enumerate(rows):
            stored = plain["trace_ids"][index, :trace_len].tolist()
            presented = shuffled["trace_ids"][index, :trace_len].tolist()
            self.assertEqual(sorted(presented), sorted(stored))
            segments = [trace[start:end] for start, end in bounds if start >= 0]
            move_index = shuffled["final_move_index"][index].item()
            final = segments[-1]
            self.assertEqual(presented[move_index - 1], FINAL)
            self.assertEqual(presented[move_index:], final[1:].tolist())
            if presented != stored:
                changed += 1
        self.assertGreater(changed, 0)

    def test_level0_presentation_is_unchanged(self):
        rows = [candidate_row(fen, candidate_entries(chess.Board(fen))) for fen in FENS]
        data = write_sequence_dataset(
            self.root / "level0", {"train": rows, "val": rows[:2]}, level=LEVEL0
        )
        positions = SequencePositions(data, "train")
        batch = [positions[index] for index in range(len(positions))]
        torch.manual_seed(0)
        shuffled = SequenceCollator(shuffle=True)(batch)
        for index, (_, trace, trace_len, _) in enumerate(batch):
            self.assertEqual(
                shuffled["trace_ids"][index, :trace_len].tolist(),
                trace[:trace_len].tolist(),
            )
            self.assertEqual(shuffled["final_move_index"][index].item(), 1)

    def test_dataset_layout_round_trips(self):
        positions = SequencePositions(self.data, "val")
        self.assertEqual(positions.level, LEVEL15)
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions.manifest["max_lines"], max_segments(LEVEL15))
        board, trace, length, bounds = positions[0]
        np.testing.assert_array_equal(board, encode_board(FENS[0], 1))
        # Two-ply PVs: three LINE segments of four tokens plus the FINAL four.
        self.assertEqual(length, 16)
        self.assertGreaterEqual(len(trace), length)
        self.assertEqual(bounds[0].tolist(), [0, 4])
        used = [row for row in bounds if row[0] >= 0]
        self.assertEqual(used[-1][1], length)
        self.assertGreater(used[-1][0], 0)
        self.assertTrue(all(end - start == 4 for start, end in used))
        for (_, end), (start, _) in zip(used, used[1:]):
            self.assertEqual(end, start)


if __name__ == "__main__":
    unittest.main()
