import tempfile
import unittest
from pathlib import Path

import chess
import torch
from teacher_release import candidate_row, play

from pondera.chess.tokens import (
    CHESS_TOKENIZER_VERSION,
    CHESS_VOCAB_SIZE,
    END,
    FINAL,
    LINE,
    MOVE,
    PAD,
    SNAPSHOT,
    VALUE,
    encode_boards,
)
from pondera.data.trace import LEVEL15, depth_snapshot, encode_trace
from pondera.models.board import pack_board_tokens
from pondera.models.checkpoint import load_model, save_checkpoint
from pondera.models.factory import create_model

CONFIG = {
    "architecture": "sequence",
    "num_blocks": 2,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_heads": 4,
    "dropout": 0.0,
}
RUY_LOPEZ = ["e2e4", "e7e5", "g1f3", "b8c6"]
# A short endgame position packs to fewer tokens than the start position.
SHORT_FEN = "8/8/8/3k4/8/3K4/8/8 w - - 0 1"


def trace_of(fen, level=LEVEL15):
    board = chess.Board(fen)
    moves = sorted(move.uci() for move in board.legal_moves)[:2]
    candidates = [(move, [move], None, 0.5) for move in moves]
    row = candidate_row(fen, candidates)
    tokens, _ = encode_trace(depth_snapshot(row), board.turn == chess.BLACK, level)
    return torch.tensor([tokens], dtype=torch.int16)


class SequencePolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.model = create_model(CONFIG).eval()
        self.ids = torch.from_numpy(encode_boards([chess.STARTING_FEN, SHORT_FEN]))

    def test_first_target_reads_the_last_board_position_without_padding_leaks(self):
        # Two independent sources for the first prediction: the cached board
        # prefill of the row alone, and the shifted trace readout of a batch
        # whose longer first row pads the second row's packed board.
        trace = torch.full((2, 8), PAD, dtype=torch.int16)
        trace[0, :7] = torch.tensor([FINAL, MOVE + 3, VALUE + 5, END, PAD, PAD, PAD])
        trace[1, :7] = torch.tensor(
            [LINE, MOVE + 8, VALUE + 2, FINAL, MOVE + 9, VALUE + 1, END]
        )
        batch = self.model.trace_logits(self.ids, trace_ids=trace)
        alone = self.model.trace_logits(self.ids[1:2], trace_ids=trace[1:2])
        torch.testing.assert_close(batch[1], alone[0])
        for index in range(len(self.ids)):
            cached, _ = self.model.prefill(self.ids[index : index + 1])
            torch.testing.assert_close(batch[index, 0], cached[0])

    def test_trace_targets_ignore_future_tokens(self):
        # Causal contract: a prediction may not depend on its own target or any
        # later token, and the first target must move with the board.
        trace = trace_of(chess.STARTING_FEN)
        perturbed = trace.clone()
        perturbed[0, 6] = END
        reference = self.model.trace_logits(self.ids[:1], trace_ids=trace)
        changed = self.model.trace_logits(self.ids[:1], trace_ids=perturbed)
        for index in range(7):
            torch.testing.assert_close(
                reference[0, index], changed[0, index], rtol=0, atol=0
            )
        self.assertFalse(torch.allclose(reference[0, 7], changed[0, 7]))
        other = torch.from_numpy(encode_boards([play(RUY_LOPEZ)[1][0]]))
        different = self.model.trace_logits(other, trace_ids=trace)
        self.assertFalse(torch.allclose(reference[0, 0], different[0, 0]))

    def test_prefill_and_decode_reproduce_the_teacher_forced_targets(self):
        trace = trace_of(chess.STARTING_FEN)
        forced = self.model.trace_logits(self.ids[:1], trace_ids=trace)
        first, cache = self.model.prefill(self.ids[:1])
        torch.testing.assert_close(first[0], forced[0, 0])
        width = pack_board_tokens(self.ids[:1])["ids"].shape[1]
        self.assertEqual(cache.prefix_length, width)
        for index in range(trace.shape[1] - 1):
            step, cache = self.model.decode(trace[:, index : index + 1], cache)
            torch.testing.assert_close(step[0], forced[0, index + 1])
        self.assertEqual(cache.length, cache.prefix_length + trace.shape[1] - 1)

    def test_generation_follows_the_greedy_teacher_forced_continuation(self):
        ids = self.ids[:1]
        logits, cache = self.model.prefill(ids)
        generated = [int(logits.argmax(-1))]
        for _ in range(5):
            step, cache = self.model.decode(
                torch.tensor([generated[-1:]], dtype=torch.int16), cache
            )
            self.assertEqual(step.shape, (1, CHESS_VOCAB_SIZE))
            generated.append(int(step.argmax(-1)))
        # Replaying the generated prefix must predict exactly those tokens.
        prefix = torch.tensor([generated[:-1]], dtype=torch.int16)
        reference = self.model.trace_logits(ids, trace_ids=prefix)
        self.assertEqual([int(row.argmax(-1)) for row in reference[0]], generated[:-1])

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequence.pth"
            trace = trace_of(chess.STARTING_FEN)
            expected = self.model.trace_logits(self.ids[:1], trace_ids=trace)
            save_checkpoint(path, self.model, CONFIG)
            loaded = load_model(path)
            actual = loaded.trace_logits(self.ids[:1], trace_ids=trace)
            torch.testing.assert_close(actual, expected)

    def test_checkpoint_before_the_snapshot_extension_masks_the_added_ids(self):
        # Vocabulary growth is append-only, so a checkpoint whose head stops at
        # the snapshot block keeps every trained output and can never select the
        # ids it never saw.
        trace = trace_of(chess.STARTING_FEN)
        expected = self.model.trace_logits(self.ids[:1], trace_ids=trace)
        state = {
            key: (value[:SNAPSHOT] if key in ("head.weight", "head.bias") else value)
            for key, value in self.model.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequence-v2.pth"
            torch.save(
                {
                    "format_version": 1,
                    "tokenizer_version": CHESS_TOKENIZER_VERSION,
                    "model_config": CONFIG,
                    "model_state_dict": state,
                },
                path,
            )
            loaded = load_model(path)
        actual = loaded.trace_logits(self.ids[:1], trace_ids=trace)
        torch.testing.assert_close(
            actual[..., :SNAPSHOT], expected[..., :SNAPSHOT], rtol=0, atol=0
        )
        self.assertTrue(torch.all(actual[..., SNAPSHOT:] < -1000))


if __name__ == "__main__":
    unittest.main()
