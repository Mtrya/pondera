import json
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch
from teacher_release import sequence_row, write_sequence_dataset

from pondera.chess.codec import decode_board, encode_board
from pondera.chess.symmetry import TOKEN_MIRROR
from pondera.chess.tracker import PathTracker
from pondera.chess.vocabulary import (
    BACK,
    BOARD,
    CHESS_TOKENIZER_VERSION,
    FINAL,
    HALFMOVE,
    IDX_TO_UCI_MOVE,
    MATE_LOSS,
    MATE_WIN,
    MOVE,
    PAD,
    REPETITION,
    RESET,
    SIDE,
    UCI_MOVE_TO_IDX,
    VALUE,
)
from pondera.data.positions import SEQUENCE_FORMAT_VERSION, SequencePositions
from pondera.data.trace import (
    ROLE_CONTROL,
    ROLE_EVAL,
    ROLE_FINAL_MOVE,
    ROLE_MOVE,
    ROLE_PAD,
    ROLE_TOOL,
    TRACE_GRAMMAR_VERSION,
    TraceError,
    TraceWriter,
    replay,
)
from pondera.models.factory import create_model
from pondera.training.sequence import SequenceCollator

# A black-to-move root reached after 1.e4, and a castling position with both
# sides holding all rights.
BLACK_TO_MOVE = chess.Board()
BLACK_TO_MOVE.push_uci("e2e4")
BLACK_TO_MOVE = BLACK_TO_MOVE.fen(en_passant="fen")
CASTLING_FEN = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"

# Frozen hand constant: value 0.5 lies in value bin 15.
BIN_OF_0_5 = 15


def move_token(uci):
    return MOVE + UCI_MOVE_TO_IDX[uci]


class ReplayTests(unittest.TestCase):
    def test_writer_program_replays_against_python_chess(self):
        writer = TraceWriter(chess.STARTING_FEN)
        writer.move("e2e4")
        writer.move("e7e5")
        writer.back()
        writer.board()
        writer.evaluate(value=0.5)
        writer.reset()
        writer.final("d2d4")

        program = replay(writer.tokens, chess.STARTING_FEN)
        self.assertEqual(
            [step.operation for step in program.steps],
            ["move", "move", "back", "board", "evaluate", "reset", "final"],
        )
        self.assertEqual(
            [step.path for step in program.steps[:3]],
            [("e2e4",), ("e2e4", "e7e5"), ("e2e4",)],
        )
        self.assertEqual(program.steps[3].path, ("e2e4",))
        self.assertEqual(program.steps[5].path, ())
        self.assertEqual(program.final_move, "d2d4")

        # python-chess is the independent source for SAN and legal sets.
        board = chess.Board(chess.STARTING_FEN)
        self.assertEqual(program.steps[0].san, board.san(chess.Move.from_uci("e2e4")))
        board.push_uci("e2e4")
        self.assertEqual(program.steps[1].san, board.san(chess.Move.from_uci("e7e5")))
        self.assertIn(
            chess.Move.from_uci(program.final_move), chess.Board().legal_moves
        )
        self.assertEqual(
            program.steps[3].observed.split()[:5],
            board.fen(en_passant="fen").split()[:5],
        )

        # Roles: the BOARD request is a control, its response is tool context,
        # and only the token after FINAL is the final move.
        board_step = program.steps[3]
        self.assertEqual(program.roles[board_step.start], ROLE_CONTROL)
        self.assertTrue(
            all(
                program.roles[index] == ROLE_TOOL
                for index in range(board_step.start + 1, board_step.end)
            )
        )
        self.assertEqual(program.roles[program.steps[0].start], ROLE_MOVE)
        self.assertEqual(program.roles[program.steps[4].start], ROLE_EVAL)
        self.assertEqual(program.roles[-2], ROLE_CONTROL)
        self.assertEqual(program.roles[-1], ROLE_FINAL_MOVE)

        rendered = program.render()
        self.assertIn("e2e4", rendered)
        self.assertIn(program.steps[3].observed, rendered)
        self.assertIn("FINAL", rendered)
        self.assertIn("d2d4", rendered)

    def test_writer_exposes_the_current_path_and_legal_moves(self):
        writer = TraceWriter(chess.STARTING_FEN)
        board = chess.Board()
        self.assertEqual(writer.path, ())
        self.assertEqual(
            writer.legal_moves(), sorted(move.uci() for move in board.legal_moves)
        )
        writer.move("e2e4")
        board.push_uci("e2e4")
        self.assertEqual(writer.path, ("e2e4",))
        self.assertEqual(
            writer.legal_moves(), sorted(move.uci() for move in board.legal_moves)
        )
        writer.back()
        self.assertEqual(writer.path, ())

    def test_reset_returns_to_the_root_from_any_depth(self):
        writer = TraceWriter(chess.STARTING_FEN)
        writer.move("e2e4")
        writer.move("e7e5")
        writer.reset()
        self.assertEqual(writer.steps[-1].path, ())
        writer.move("d2d4")
        writer.final("g1f3")
        program = replay(writer.tokens, chess.STARTING_FEN)
        self.assertEqual(program.steps[2].operation, "reset")
        self.assertEqual(program.steps[3].path, ("d2d4",))
        self.assertEqual(program.final_move, "g1f3")

        # RESET at the root is a valid no-op control.
        writer = TraceWriter(chess.STARTING_FEN)
        writer.reset()
        writer.final("e2e4")
        program = replay(writer.tokens, chess.STARTING_FEN)
        self.assertEqual(program.steps[0].path, ())

    def test_back_at_the_root_is_illegal(self):
        writer = TraceWriter(chess.STARTING_FEN)
        with self.assertRaises(ValueError):
            writer.back()
        tokens = [BACK, FINAL, move_token("e2e4")]
        with self.assertRaises(TraceError) as context:
            replay(tokens, chess.STARTING_FEN)
        self.assertEqual(context.exception.tag, "back_at_root")

        # BACK is legal after a move and yields the parent path.
        writer.move("e2e4")
        writer.back()
        writer.final("d2d4")
        program = replay(writer.tokens, chess.STARTING_FEN)
        self.assertEqual(program.steps[-2].path, ())

    def test_final_uses_the_root_legal_set_not_the_current_node(self):
        writer = TraceWriter(chess.STARTING_FEN)
        writer.move("e2e4")
        # d2d4 is legal at the root but illegal at the current black-to-move node.
        writer.final("d2d4")
        program = replay(writer.tokens, chess.STARTING_FEN)
        self.assertEqual(program.final_move, "d2d4")

        # e7e5 is legal at the current node but not at the root.
        tokens = [move_token("e2e4"), FINAL, move_token("e7e5")]
        with self.assertRaises(TraceError) as context:
            replay(tokens, chess.STARTING_FEN)
        self.assertEqual(context.exception.tag, "final_illegal")

        with self.assertRaises(ValueError):
            TraceWriter(chess.STARTING_FEN).final("e7e5")

    def test_final_terminates_and_rejects_trailing_tokens(self):
        writer = TraceWriter(chess.STARTING_FEN)
        writer.final("e2e4")
        self.assertTrue(writer.finished)
        for operation in (
            lambda: writer.move("d2d4"),
            writer.back,
            writer.reset,
            lambda: writer.evaluate(value=0.1),
            writer.board,
            lambda: writer.final("d2d4"),
        ):
            with self.assertRaises(ValueError):
                operation()

        tokens = [FINAL, move_token("e2e4")]
        program = replay([*tokens, PAD, PAD], chess.STARTING_FEN)
        self.assertEqual(program.tokens, tuple(tokens))
        for trailing in (BACK, RESET, BOARD, VALUE + 3):
            with self.subTest(trailing=trailing):
                with self.assertRaises(TraceError) as context:
                    replay([*tokens, trailing], chess.STARTING_FEN)
                self.assertEqual(context.exception.tag, "trailing_tokens")

        for tokens in ([FINAL], [FINAL, VALUE + 3]):
            with self.assertRaises(TraceError) as context:
                replay(tokens, chess.STARTING_FEN)
            self.assertEqual(context.exception.tag, "final_without_move")

    def test_malformed_programs_report_their_first_broken_rule(self):
        cases = {
            "missing_final": [move_token("e2e4")],
            "unexpected_token": [SIDE + 1],
            "pad_inside_trace": [move_token("e2e4"), PAD, FINAL, move_token("d2d4")],
        }
        for expected, tokens in cases.items():
            with self.subTest(error=expected):
                with self.assertRaises(TraceError) as context:
                    replay(tokens, chess.STARTING_FEN)
                self.assertEqual(context.exception.tag, expected)

    def test_claim_draw_is_not_a_legal_move(self):
        # The pseudo-move stays in the action vocabulary but never passes rules.
        self.assertIn("<claim_draw>", UCI_MOVE_TO_IDX)
        writer = TraceWriter(chess.STARTING_FEN)
        with self.assertRaises(ValueError):
            writer.move("<claim_draw>")
        with self.assertRaises(ValueError):
            writer.final("<claim_draw>")
        tokens = [move_token("<claim_draw>"), FINAL, move_token("e2e4")]
        with self.assertRaises(TraceError) as context:
            replay(tokens, chess.STARTING_FEN)
        self.assertEqual(context.exception.tag, "illegal_move")

    def test_evaluations_use_root_perspective_tokens(self):
        writer = TraceWriter(BLACK_TO_MOVE)
        writer.evaluate(value=0.5)
        writer.evaluate(mate=3, win=True)
        writer.evaluate(mate=2, win=False)
        writer.reset()
        writer.final("c7c5")

        program = replay(writer.tokens, BLACK_TO_MOVE)
        self.assertEqual(writer.tokens[0], VALUE + BIN_OF_0_5)
        self.assertEqual(writer.tokens[1], MATE_WIN + 3)
        self.assertEqual(writer.tokens[2], MATE_LOSS + 2)
        self.assertEqual((program.steps[1].mate, program.steps[1].mate_win), (3, True))
        self.assertEqual((program.steps[2].mate, program.steps[2].mate_win), (2, False))
        self.assertGreater(program.steps[0].evaluation, 0)
        # Distance zero still distinguishes a win atom from a loss atom.
        writer = TraceWriter(BLACK_TO_MOVE)
        writer.evaluate(mate=0, win=True)
        writer.evaluate(mate=0, win=False)
        writer.final("c7c5")
        program = replay(writer.tokens, BLACK_TO_MOVE)
        self.assertEqual(writer.tokens[0], MATE_WIN)
        self.assertEqual(writer.tokens[1], MATE_LOSS)
        self.assertEqual(
            [(step.mate, step.mate_win) for step in program.steps[:2]],
            [(0, True), (0, False)],
        )

    def test_board_observations_match_the_true_position(self):
        writer = TraceWriter(CASTLING_FEN)
        writer.move("e1g1")
        writer.board()
        writer.final("a1b1")
        program = replay(writer.tokens, CASTLING_FEN)
        board = chess.Board(CASTLING_FEN)
        board.push_uci("e1g1")
        step = program.steps[1]
        self.assertEqual(
            step.observed.split()[:5], board.fen(en_passant="fen").split()[:5]
        )
        self.assertEqual(step.repetition, 1)
        decoded, repetition = decode_board(
            np.asarray(writer.tokens[step.start + 1 : step.end])
        )
        self.assertEqual(decoded.split()[:5], board.fen(en_passant="fen").split()[:5])
        self.assertEqual(repetition, 1)

    def test_board_observation_saturates_large_halfmove_clock(self):
        root_fen = "7k/8/8/8/8/8/8/K6R w - - 99 1"
        writer = TraceWriter(root_fen)
        writer.move("h1h2")
        writer.move("h8g8")
        writer.board()
        writer.final("h1h2")
        board = chess.Board(root_fen)
        board.push_uci("h1h2")
        board.push_uci("h8g8")
        self.assertEqual(board.halfmove_clock, 101)

        program = replay(writer.tokens, root_fen)
        step = program.steps[2]
        self.assertEqual(step.position, board.fen(en_passant="fen"))
        self.assertEqual(step.observed.split()[:4], board.fen().split()[:4])
        self.assertEqual(step.observed.split()[4], "100")
        self.assertEqual(writer.tokens[step.end - 2], HALFMOVE + 100)

    def test_board_observation_rejects_wrong_or_broken_content(self):
        writer = TraceWriter(chess.STARTING_FEN)
        writer.move("e2e4")
        writer.board()
        writer.final("d2d4")
        step = writer.steps[1]
        tokens = list(writer.tokens)

        wrong = list(tokens)
        wrong[step.start + 1] = 0  # a1 pawn atom, not the observed position
        with self.assertRaises(TraceError) as context:
            replay(wrong, chess.STARTING_FEN)
        self.assertEqual(context.exception.tag, "board_mismatch")

        truncated = tokens[: step.end - 1]
        with self.assertRaises(TraceError) as context:
            replay(truncated, chess.STARTING_FEN)
        self.assertEqual(context.exception.tag, "board_truncated")

        invalid = list(tokens)
        invalid[step.end - 7] = SIDE  # duplicate side atom
        with self.assertRaises(TraceError) as context:
            replay(invalid, chess.STARTING_FEN)
        self.assertEqual(context.exception.tag, "board_invalid")

        # The observation carries the true halfmove clock and repetition count.
        for offset, token in (
            (step.end - 2, HALFMOVE + 7),  # a clock the position does not have
            (step.end - 1, REPETITION + 1),  # claims a second occurrence
        ):
            with self.subTest(offset=offset):
                wrong_counter = list(tokens)
                wrong_counter[offset] = token
                with self.assertRaises(TraceError) as context:
                    replay(wrong_counter, chess.STARTING_FEN)
                self.assertEqual(context.exception.tag, "board_mismatch")

    def test_board_atoms_are_a_set_not_a_byte_sequence(self):
        # Replay validates by decoding the observation, so piece atoms may
        # arrive in any order and still describe the same position.
        writer = TraceWriter(chess.STARTING_FEN)
        writer.move("e2e4")
        writer.board()
        writer.final("d2d4")
        step = writer.steps[1]
        tokens = list(writer.tokens)
        pieces = tokens[step.start + 1 : step.end - 8]
        reordered = (
            tokens[: step.start + 1] + list(reversed(pieces)) + tokens[step.end - 8 :]
        )
        self.assertNotEqual(reordered, tokens)
        program = replay(reordered, chess.STARTING_FEN)
        self.assertEqual(program.steps[1].observed, writer.steps[1].observed)


class BlackRootMirrorTests(unittest.TestCase):
    def test_board_response_mirrors_pieces_but_not_side_or_counters(self):
        root = chess.Board()
        for move in ("e2e4", "e7e5", "g1f3"):
            root.push_uci(move)
        root_fen = root.fen(en_passant="fen")
        self.assertEqual(root.turn, chess.BLACK)
        self.assertEqual(root.halfmove_clock, 1)

        writer = TraceWriter(root_fen)
        self.assertTrue(writer.mirrored)
        writer.move("b8c6")  # odd ply: White to move again, clock 2
        writer.board()
        writer.final("d7d5")
        program = replay(writer.tokens, root_fen)

        true_board = chess.Board(root_fen)
        true_board.push_uci("b8c6")
        self.assertEqual(true_board.turn, chess.WHITE)
        self.assertEqual(true_board.halfmove_clock, 2)
        step = program.steps[1]
        self.assertEqual(
            step.observed.split()[:5], true_board.fen(en_passant="fen").split()[:5]
        )
        # The stored side atom names the true side, not the mirrored one.
        self.assertEqual(step.observed.split()[1], "w")

        stored = writer.tokens[step.start + 1 : step.end]
        directly, _ = decode_board(stored)
        self.assertEqual(directly.split()[1], "w")
        self.assertNotEqual(directly.split()[0], true_board.board_fen())
        absolute, _ = decode_board(np.asarray(TOKEN_MIRROR)[stored])
        self.assertEqual(
            absolute.split()[:5], true_board.fen(en_passant="fen").split()[:5]
        )
        # A naive Board.mirror() would flip the side to move; the response does not.
        self.assertEqual(true_board.mirror().fen().split()[1], "b")

    def test_black_root_moves_are_stored_reflected(self):
        writer = TraceWriter(BLACK_TO_MOVE)
        writer.move("e7e5")
        writer.back()
        writer.final("c7c5")
        program = replay(writer.tokens, BLACK_TO_MOVE)
        self.assertEqual(program.steps[0].path, ("e7e5",))
        self.assertEqual(program.steps[0].uci, "e7e5")
        self.assertEqual(program.steps[0].san, "e5")
        self.assertEqual(program.final_move, "c7c5")
        # The stored move is the rank reflection of the absolute move.
        self.assertEqual(
            IDX_TO_UCI_MOVE[writer.tokens[0] - MOVE],
            "e2e4",
        )

    def test_root_repetition_reaches_the_observation(self):
        tracker = PathTracker(chess.STARTING_FEN, repetition=3)
        decoded, repetition = decode_board(tracker.board_tokens())
        self.assertEqual(repetition, 3)
        self.assertEqual(decoded.split()[:5], chess.Board().fen().split()[:5])
        tracker.push("g1f3")
        _, repetition = decode_board(tracker.board_tokens())
        self.assertEqual(repetition, 1)


class DatasetContractTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

        def row(fen, repetition=1):
            board = chess.Board(fen)
            legal = sorted(move.uci() for move in board.legal_moves)
            writer = TraceWriter(fen, repetition)
            writer.board()
            writer.move(legal[0])
            writer.evaluate(value=0.25)
            writer.back()
            writer.final(legal[1])
            return sequence_row(fen, writer.tokens, writer.roles, repetition)

        fens = [chess.STARTING_FEN, BLACK_TO_MOVE, CASTLING_FEN]
        self.rows = [row(fen) for fen in fens]
        self.data = write_sequence_dataset(
            self.root / "data", {"train": self.rows, "val": self.rows[:2]}
        )

    def test_manifest_carries_the_grammar_and_not_old_level_fields(self):
        manifest = json.loads((self.data / "manifest.json").read_text())
        self.assertEqual(manifest["kind"], "teacher-sequence")
        self.assertEqual(manifest["grammar_version"], 1)
        self.assertEqual(manifest["format_version"], 2)
        for retired in ("level", "line_bounds", "max_lines", "pv_limit"):
            self.assertNotIn(retired, manifest)

    def test_dataset_rejects_a_missing_grammar_version(self):
        manifest_path = self.data / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        del manifest["grammar_version"]
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            SequencePositions(self.data, "train")

    def test_dataset_arrays_and_roles(self):
        positions = SequencePositions(self.data, "train")
        self.assertEqual(len(positions), 3)
        boards, trace, length, roles = positions[0]
        self.assertEqual(len(trace), positions.trace.shape[1])
        self.assertEqual(int(length), len(self.rows[0]["trace"]))
        self.assertEqual(trace[:length].tolist(), self.rows[0]["trace"].tolist())
        self.assertEqual(roles[:length].tolist(), self.rows[0]["roles"].tolist())
        self.assertTrue(np.all(roles[length:] == ROLE_PAD))
        self.assertEqual(int(roles[length - 1]), ROLE_FINAL_MOVE)

    def test_collator_pads_tokens_and_roles_and_locates_the_final_move(self):
        positions = SequencePositions(self.data, "train")
        rows = [positions[index] for index in range(len(positions))]
        batch = SequenceCollator()(rows)
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
            self.assertEqual(int(batch["final_move_index"][index]), length - 1)
            self.assertEqual(
                int(batch["trace_roles"][index, length - 1]), ROLE_FINAL_MOVE
            )


def write_shard(directory, index, fen):
    """Publish one one-row shard whose board array is exactly its own width."""
    board = chess.Board(fen)
    writer = TraceWriter(fen)
    writer.final(sorted(move.uci() for move in board.legal_moves)[0])
    shard = directory / f"shard-{index:06d}"
    shard.mkdir()
    np.save(shard / "board_tokens.npy", encode_board(fen)[None, :])
    np.save(shard / "trace_tokens.npy", writer.tokens[None, :])
    np.save(shard / "trace_len.npy", np.asarray([len(writer.tokens)], dtype=np.int16))
    np.save(shard / "trace_roles.npy", writer.roles[None, :])
    for name in (
        "state_no_tool",
        "state_tool",
        "state_best",
        "state_basis",
        "state_value",
    ):
        np.save(shard / f"{name}.npy", np.zeros((1, 0), dtype=np.int16))
    np.save(shard / "state_count.npy", np.zeros(1, dtype=np.int16))
    (shard / "roots.jsonl").write_text(json.dumps({"split": "train"}) + "\n")
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
                        "kind": "teacher-sequence",
                        "format_version": SEQUENCE_FORMAT_VERSION,
                        "grammar_version": TRACE_GRAMMAR_VERSION,
                        "tokenizer_version": CHESS_TOKENIZER_VERSION,
                        "shards": [{"path": path, "rows": 1} for path in paths],
                    }
                )
            )
            positions = SequencePositions(root, "train")
            rows = [positions[0], positions[1]]
            widths = [len(row[0]) for row in rows]
            self.assertLess(widths[0], widths[1])

            batch = SequenceCollator()(rows)
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
                    "architecture": "sequence",
                    "num_blocks": 1,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_heads": 2,
                }
            ).eval()
            with torch.no_grad():
                batched = model.trace_logits(batch["ids"], batch["trace_ids"])
                for index, row in enumerate(rows):
                    alone_batch = SequenceCollator()([row])
                    alone = model.trace_logits(
                        alone_batch["ids"], alone_batch["trace_ids"]
                    )
                    torch.testing.assert_close(batched[index], alone[0])


if __name__ == "__main__":
    unittest.main()
