import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch

from pondera.chess.codec import encode_boards
from pondera.chess.vocabulary import CHESS_VOCAB_SIZE, PAD, RESET
from pondera.data.trace import TraceWriter
from pondera.models.checkpoint import load_model, save_checkpoint
from pondera.models.factory import create_model
from pondera.training.cost import sequence_flops

CONFIG = {
    "architecture": "sequence",
    "num_blocks": 2,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_heads": 4,
    "dropout": 0.0,
}
# A short endgame position packs to fewer tokens than the start position.
SHORT_FEN = "8/8/8/3k4/8/3K4/8/8 w - - 0 1"


def trace_of(fen):
    board = chess.Board(fen)
    legal = sorted(move.uci() for move in board.legal_moves)
    writer = TraceWriter(fen)
    writer.move(legal[0])
    writer.evaluate(value=0.5)
    writer.back()
    writer.final(legal[1])
    return torch.from_numpy(np.stack([writer.tokens]))


def final_trace(fen):
    board = chess.Board(fen)
    move = sorted(move.uci() for move in board.legal_moves)[0]
    writer = TraceWriter(fen)
    writer.final(move)
    return torch.from_numpy(np.stack([writer.tokens]))


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
        first = trace_of(chess.STARTING_FEN)[0]
        second = final_trace(SHORT_FEN)[0]
        trace[0, : len(first)] = first
        trace[1, : len(second)] = second
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
        self.assertEqual(trace.shape[1], 5)
        perturbed = trace.clone()
        perturbed[0, 2] = RESET
        reference = self.model.trace_logits(self.ids[:1], trace_ids=trace)
        changed = self.model.trace_logits(self.ids[:1], trace_ids=perturbed)
        for index in range(3):
            torch.testing.assert_close(
                reference[0, index], changed[0, index], rtol=0, atol=0
            )
        self.assertFalse(torch.allclose(reference[0, 3], changed[0, 3]))
        other = torch.from_numpy(encode_boards([SHORT_FEN]))
        different = self.model.trace_logits(other, trace_ids=trace)
        self.assertFalse(torch.allclose(reference[0, 0], different[0, 0]))

    def test_prefix_is_a_position_free_set(self):
        # The prefix carries no positional term: permuting its atoms cannot
        # change any trace prediction.
        trace = trace_of(chess.STARTING_FEN)
        reference = self.model.trace_logits(self.ids[:1], trace_ids=trace)
        order = torch.randperm(self.ids.shape[1])
        permuted = self.model.trace_logits(self.ids[:1, order], trace_ids=trace)
        torch.testing.assert_close(permuted, reference)

    def test_prefill_and_decode_reproduce_the_teacher_forced_targets(self):
        trace = trace_of(chess.STARTING_FEN)
        forced = self.model.trace_logits(self.ids[:1], trace_ids=trace)
        first, cache = self.model.prefill(self.ids[:1])
        torch.testing.assert_close(first[0], forced[0, 0])
        width = self.ids.shape[1]
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


class SequenceFlopsTests(unittest.TestCase):
    def test_hand_derived_literal(self):
        # Two layers, d=16, f=32, B=10, T=3, V=3006. Linear per layer:
        # 2*13*(4*16^2 + 3*16*32) = 66560. Attention pairs:
        # 10^2 + 10*3 + 3*4/2 = 136, costing 4*16*136 = 8704.
        # Per layer 75264, two layers 150528. Head: 2*3*16*3006 = 288576.
        # Forward 439104; training counts forward plus two backward passes.
        config = {
            "architecture": "sequence",
            "num_blocks": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
        }
        self.assertEqual(float(sequence_flops(config, [10], [3])[0]), 1317312.0)
        self.assertEqual(
            float(sequence_flops(config, [10], [3], training=False)[0]), 439104.0
        )

    def test_rejects_other_architectures_and_empty_rows(self):
        config = {
            "architecture": "sequence",
            "num_blocks": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
        }
        with self.assertRaises(ValueError):
            sequence_flops({**config, "architecture": "depth"}, [10], [3])
        with self.assertRaises(ValueError):
            sequence_flops(config, [10], [0])


if __name__ == "__main__":
    unittest.main()
