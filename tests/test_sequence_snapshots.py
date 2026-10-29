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

from pondera.chess.mapping import IDX_TO_UCI_MOVE
from pondera.chess.tokens import (
    BOARD_PIECES,
    BOARD_TOKENS,
    CASTLING,
    CHESS_VOCAB_SIZE,
    EN_PASSANT,
    END,
    FINAL,
    LINE,
    MOVE,
    PAD,
    SIDE,
    VALUE,
    encode_board,
)
from pondera.data.positions import SequencePositions
from pondera.data.trace import (
    S2E,
    S2R3,
    canonical_board,
    decode_snapshot,
    depth_snapshot,
    encode_trace,
    is_board_token,
    max_segments,
    parse_trace,
    snapshot_of,
    snapshot_pieces,
    snapshot_positions,
)
from pondera.models.factory import create_model
from pondera.training.sequence import SequenceCollator, trace_loss

RUY_LOPEZ = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]
BLACK_TO_MOVE = play(RUY_LOPEZ)[1][0]
CASTLING_FEN = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
FENS = [chess.STARTING_FEN, BLACK_TO_MOVE, CASTLING_FEN]


def truth_castling(board):
    return (
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
    )


def line_segments(tokens):
    segments, error = parse_trace(tokens)
    assert error is None, (error, tokens)
    return segments


class BoardAtomTests(unittest.TestCase):
    def test_board_atoms_are_joint_square_piece_ids(self):
        seen = set()
        for square in range(64):
            for piece in range(12):
                token = square * BOARD_PIECES + piece
                self.assertTrue(is_board_token(token))
                self.assertEqual(divmod(token, BOARD_PIECES), (square, piece))
                seen.add(token)
        self.assertEqual(len(seen), 12 * 64)
        self.assertEqual((min(seen), max(seen)), (0, BOARD_TOKENS - 1))
        for token in (BOARD_TOKENS, SIDE, MOVE, PAD):
            with self.subTest(token=token):
                self.assertFalse(is_board_token(token))

    def test_reading_a_board_and_writing_a_snapshot_emit_the_same_atoms(self):
        # White to move: the trace snapshot is not mirrored, so its piece run
        # must be exactly the prefix's piece run; only the counters differ.
        board = chess.Board(chess.STARTING_FEN)
        prefix = encode_board(board.fen(), 1)
        run = snapshot_of(board, False)
        pieces = len(snapshot_pieces(board))
        self.assertEqual(run[:pieces], prefix[:pieces].tolist())
        self.assertEqual(run[pieces], SIDE)
        self.assertEqual(len(run), pieces + 6)


class SnapshotEncodingTests(unittest.TestCase):
    def test_snapshot_run_matches_python_chess(self):
        for fen in FENS:
            with self.subTest(fen=fen):
                board = chess.Board(fen)
                decoded = decode_snapshot(snapshot_of(board, mirror=False))
                self.assertEqual(dict(decoded.pieces), dict(snapshot_pieces(board)))
                self.assertEqual(decoded.side_black, board.turn == chess.BLACK)
                self.assertEqual(decoded.castling, truth_castling(board))
                self.assertEqual(decoded.ep, board.ep_square or 64)

    def test_mirrored_snapshot_keeps_the_root_frame_side(self):
        board = chess.Board(BLACK_TO_MOVE)
        decoded = decode_snapshot(snapshot_of(board, mirror=True))
        self.assertEqual(
            dict(decoded.pieces), dict(snapshot_pieces(canonical_board(board, True)))
        )
        self.assertEqual(dict(decoded.pieces), dict(snapshot_pieces(board.mirror())))
        # The mirror swaps the drawn colors but not the root-frame side atom.
        self.assertEqual(decoded.side_black, board.turn == chess.BLACK)

    def test_snapshots_follow_the_replayed_pv_in_the_move_frame(self):
        for fen in FENS:
            with self.subTest(fen=fen):
                board = chess.Board(fen)
                black = board.turn == chess.BLACK
                row = candidate_row(fen, candidate_entries(board))
                tokens, _ = encode_trace(depth_snapshot(row), black, S2R3, board=board)
                line = line_segments(tokens)[0]
                self.assertTrue(line.snapshots)
                replay = canonical_board(board, black)
                for number, token in enumerate(line.moves, 1):
                    replay.push_uci(IDX_TO_UCI_MOVE[token - MOVE])
                    for moves_before, run in line.snapshots:
                        if moves_before != number:
                            continue
                        decoded = decode_snapshot(run)
                        self.assertEqual(
                            dict(decoded.pieces), dict(snapshot_pieces(replay))
                        )
                        self.assertEqual(decoded.castling, truth_castling(replay))
                        self.assertEqual(decoded.ep, replay.ep_square or 64)
                        # The side atom names the mover in the root frame and
                        # flips with every ply of the replayed line.
                        self.assertEqual(decoded.side_black, black ^ (number % 2 == 1))

    def test_snapshot_cadence_matches_the_level_contract(self):
        board = chess.Board(chess.STARTING_FEN)
        for level, expected in (
            (S2E, {10: [10], 4: [4]}),
            (S2R3, {10: [3, 6, 9, 10], 4: [3, 4]}),
        ):
            for plies in sorted(expected):
                with self.subTest(level=level, plies=plies):
                    rows = candidate_row(
                        chess.STARTING_FEN,
                        candidate_entries(board, count=1, plies=plies),
                    )
                    tokens, _ = encode_trace(
                        depth_snapshot(rows), False, level, board=board
                    )
                    segments = line_segments(tokens)
                    line = segments[0]
                    self.assertEqual(
                        [moves_before for moves_before, _ in line.snapshots],
                        expected[plies],
                    )
                    for moves_before, run in line.snapshots:
                        self.assertLessEqual(moves_before, plies)
                        decode_snapshot(run)
                    # The answer carries no snapshot at any level.
                    self.assertEqual(segments[-1].snapshots, ())

    def test_snapshot_metadata_tracks_castling_and_en_passant(self):
        board = chess.Board(chess.STARTING_FEN)
        entry = [("e2e4", ["e2e4"], None, 0.5)]
        tokens, _ = encode_trace(
            depth_snapshot(candidate_row(chess.STARTING_FEN, entry)),
            False,
            S2E,
            board=board,
        )
        snapshot = decode_snapshot(line_segments(tokens)[0].snapshots[0][1])
        self.assertEqual(snapshot.castling, (1, 1, 1, 1))
        self.assertEqual(snapshot.ep, chess.E3)

        # Castling rights follow the replay: white loses both by castling and
        # black loses the queenside right by moving the a8 rook.
        board = chess.Board(CASTLING_FEN)
        entry = [("e1g1", ["e1g1", "a8b8"], None, 0.5)]
        tokens, _ = encode_trace(
            depth_snapshot(candidate_row(CASTLING_FEN, entry)), False, S2E, board=board
        )
        snapshot = decode_snapshot(line_segments(tokens)[0].snapshots[0][1])
        self.assertEqual(snapshot.castling, (0, 0, 1, 0))
        self.assertEqual(snapshot.ep, 64)
        replay = chess.Board(CASTLING_FEN)
        for move in ("e1g1", "a8b8"):
            replay.push_uci(move)
        self.assertEqual(snapshot.castling, truth_castling(replay))
        self.assertEqual(
            dict(snapshot.pieces),
            dict(snapshot_pieces(canonical_board(replay, False))),
        )

    def test_parse_reports_broken_snapshot_runs(self):
        move, evaluation = MOVE + 7, VALUE + 5
        first = 10 * BOARD_PIECES
        second = 20 * BOARD_PIECES + 6
        metadata = [
            SIDE,
            CASTLING + 1,
            CASTLING + 2,
            CASTLING + 4,
            CASTLING + 6,
            EN_PASSANT + 64,
        ]
        answer = [FINAL, move, evaluation, END]
        _, error = parse_trace(
            [LINE, move, first, second, *metadata, evaluation, *answer]
        )
        self.assertIsNone(error)
        cases = [
            ("snapshot_square_order", [second, first, *metadata]),
            ("snapshot_truncated", [first, second, *metadata[:-1]]),
            (
                "snapshot_truncated",
                [first, second, *metadata[:-1], EN_PASSANT + 65],
            ),
            ("snapshot_side", [first, SIDE + 2, *metadata[1:]]),
            ("snapshot_castling", [first, SIDE, CASTLING + 3, *metadata[2:]]),
        ]
        for expected, body in cases:
            with self.subTest(error=expected, body=body[:3]):
                _, error = parse_trace([LINE, move, *body, evaluation])
                self.assertEqual(error, expected)
        # Snapshots belong to LINE segments; an answer carries none.
        _, error = parse_trace([FINAL, first, *metadata, move, evaluation, END])
        self.assertEqual(error, "final_without_move")
        _, error = parse_trace([LINE, first, *metadata, evaluation])
        self.assertEqual(error, "line_without_move")

    def test_snapshot_positions_and_encode_guards(self):
        self.assertEqual(max_segments(S2E), max_segments(S2R3))
        self.assertEqual(max_segments(S2E), 5)
        self.assertEqual(snapshot_positions(S2E, 0), ())
        self.assertEqual(snapshot_positions(S2R3, 9), (3, 6, 9))
        self.assertEqual(snapshot_positions(S2R3, 3), (3,))
        row = candidate_row(chess.STARTING_FEN, candidate_entries(chess.Board()))
        with self.assertRaises(ValueError):
            encode_trace(depth_snapshot(row), False, S2E)
        with self.assertRaises(ValueError):
            encode_trace(depth_snapshot(row), False, "level2")


class SnapshotOrderTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        rows = []
        for fen in FENS:
            board = chess.Board(fen)
            for plies in (2, 4, 5, 7):
                rows.append(candidate_row(fen, candidate_entries(board, plies=plies)))
        self.data = write_sequence_dataset(
            self.root, {"train": rows, "val": rows[:2]}, level=S2R3
        )

    def test_collator_moves_whole_lines_and_keeps_snapshots_inside(self):
        positions = SequencePositions(self.data, "train")
        rows = [positions[index] for index in range(len(positions))]
        plain = SequenceCollator(shuffle=False)(rows)
        shuffled = SequenceCollator(shuffle=True, salt=640)(rows)
        changed = 0
        for index, (_, trace, trace_len, bounds) in enumerate(rows):
            stored = plain["trace_ids"][index, :trace_len].tolist()
            presented = shuffled["trace_ids"][index, :trace_len].tolist()
            self.assertEqual(sorted(presented), sorted(stored))
            segments = [
                [int(value) for value in trace[start:end]]
                for start, end in bounds
                if start >= 0
            ]
            for start, end in bounds:
                if start >= 0 and trace[start] == LINE:
                    self.assertGreater(end - start, 4)
                    self.assertTrue(
                        any(is_board_token(token) for token in trace[start:end])
                    )
            # Whole lines move together: the presented trace is a concatenation
            # of the stored segments with the FINAL segment last.
            walk, remaining, order = 0, list(segments), []
            while walk < len(presented):
                matched = next(
                    index
                    for index, segment in enumerate(remaining)
                    if presented[walk : walk + len(segment)] == segment
                )
                segment = remaining.pop(matched)
                order.append(segment)
                walk += len(segment)
            self.assertEqual(remaining, [])
            self.assertEqual(order[-1], segments[-1])
            if presented != stored:
                changed += 1
        self.assertGreater(changed, 0)

    def test_model_reads_snapshot_traces(self):
        torch.set_num_threads(1)
        model = create_model(
            {
                "architecture": "sequence",
                "num_blocks": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 4,
                "dropout": 0.0,
            }
        ).eval()
        board, trace, length, _ = SequencePositions(self.data, "val")[0]
        self.assertEqual(model.token_embedding.num_embeddings, CHESS_VOCAB_SIZE)
        stored = trace[:length]
        self.assertGreater(sum(is_board_token(int(token)) for token in stored), 0)
        ids = torch.from_numpy(np.stack([board, board]))
        trace_ids = torch.from_numpy(np.stack([stored, stored])).to(torch.int16)
        logits = model(ids=ids, trace_ids=trace_ids)
        width = ids.shape[1]
        self.assertEqual(logits.shape, (2, width + length, CHESS_VOCAB_SIZE))
        segments, error = parse_trace(stored.tolist())
        self.assertIsNone(error)
        self.assertTrue(all(segment.snapshots for segment in segments[:-1]))

    def test_snapshot_traces_train_and_generate_through_one_table(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        model = create_model(
            {
                "architecture": "sequence",
                "num_blocks": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 4,
                "dropout": 0.0,
            }
        )
        row = SequencePositions(self.data, "val")[0]
        batch = SequenceCollator(shuffle=False)([row])
        loss, _, mask = trace_loss(model, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(int(mask.sum()), 0)
        loss.backward()
        grad = model.token_embedding.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.norm()), 0)
        # The cached generation path walks the same snapshot atoms: prefill plus
        # one-token decode must reproduce the teacher-forced logits exactly.
        model.eval()
        trace = batch["trace_ids"][:1]
        length = int((trace != PAD).sum())
        self.assertGreater(sum(is_board_token(int(token)) for token in trace[0]), 0)
        forced = model.trace_logits(batch["ids"][:1], trace[:, :length])
        first, cache = model.prefill(batch["ids"][:1])
        torch.testing.assert_close(first[0], forced[0, 0])
        for index in range(length - 1):
            step, cache = model.decode(trace[:, index : index + 1], cache)
            torch.testing.assert_close(step[0], forced[0, index + 1])


if __name__ == "__main__":
    unittest.main()
