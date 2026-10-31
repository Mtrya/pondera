import math
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch
from torch.nn import functional as F

from pondera.chess.codec import encode_boards
from pondera.chess.vocabulary import (
    BACK,
    BOARD,
    FINAL,
    MOVE,
    PAD,
    RESET,
    UCI_MOVE_TO_IDX,
    VALUE,
)
from pondera.data.trace import TraceWriter
from pondera.inference.sequence import (
    STOP_BUDGET,
    STOP_CAP,
    STOP_FINAL,
    GeneratedTrace,
    RootPosition,
    assess,
    autonomous,
    budget_grid,
    budget_points,
    fixed_budget,
    generate,
    legal_tokens,
    model_token_count,
    move_log_probability,
    no_tool_length,
    no_tool_token_count,
    p95_trace_length,
    trace_length_budget,
)
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
SHORT_FEN = "8/8/8/3k4/8/3K4/8/8 w - - 0 1"
BLACK_TO_MOVE = chess.Board()
BLACK_TO_MOVE.push_uci("e2e4")
BLACK_TO_MOVE = BLACK_TO_MOVE.fen(en_passant="fen")


def move_token(uci):
    return MOVE + UCI_MOVE_TO_IDX[uci]


def memorized_trace(fen):
    """A scripted thought with BACK, RESET, an evaluation, and a root decision."""
    board = chess.Board(fen)
    legal = sorted(move.uci() for move in board.legal_moves)
    writer = TraceWriter(fen)
    writer.move(legal[0])
    writer.back()
    writer.reset()
    writer.evaluate(value=0.5)
    writer.move(legal[2])
    writer.final(legal[3])
    return writer


class GenerationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.model = create_model(CONFIG).eval()

    @staticmethod
    def memorize(model, fen, trace):
        """Fit a tiny real model to one trace so greedy generation is scripted."""
        ids = torch.from_numpy(encode_boards([fen]))
        trace_ids = torch.from_numpy(np.asarray([trace], dtype=np.int16))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        model.train()
        for _ in range(200):
            logits = model.trace_logits(ids, trace_ids)
            loss = F.cross_entropy(logits[0].float(), trace_ids[0].long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()

    def test_legal_tokens_follow_the_real_rule_state(self):
        writer = TraceWriter(chess.STARTING_FEN)
        allowed = set(legal_tokens(writer, final_allowed=False))
        board = chess.Board(chess.STARTING_FEN)
        legal = {move_token(move.uci()) for move in board.legal_moves}
        self.assertEqual(allowed & set(range(MOVE, VALUE)), legal)
        self.assertTrue(set(range(VALUE, PAD)) <= allowed)
        self.assertIn(RESET, allowed)
        self.assertNotIn(BACK, allowed)
        self.assertNotIn(FINAL, allowed)
        self.assertNotIn(BOARD, allowed)
        self.assertNotIn(PAD, allowed)
        self.assertNotIn(move_token("<claim_draw>"), allowed)

        first = sorted(move.uci() for move in board.legal_moves)[0]
        writer.move(first)
        board.push_uci(first)
        allowed = set(legal_tokens(writer, final_allowed=True))
        self.assertIn(BACK, allowed)
        self.assertIn(FINAL, allowed)
        self.assertEqual(
            allowed & set(range(MOVE, VALUE)),
            {move_token(move.uci()) for move in board.legal_moves},
        )

    def test_black_to_move_legal_tokens_use_the_canonical_frame(self):
        writer = TraceWriter(BLACK_TO_MOVE)
        allowed = set(legal_tokens(writer, final_allowed=False))
        # Hand-derived rank mirror: c7->c2 and c5->c4, so c7c5 is stored as c2c4.
        self.assertIn(move_token("c2c4"), allowed)
        self.assertNotIn(move_token("c7c5"), allowed)
        writer.move("c7c5")
        self.assertEqual(int(writer.tokens[-1]), move_token("c2c4"))
        self.assertTrue(writer.mirrored)

    def test_fixed_budget_masks_final_and_counts_only_thought_tokens(self):
        root = RootPosition(chess.STARTING_FEN)
        direct = generate(self.model, [root], fixed_budget(0))[0]
        self.assertEqual(direct.thinking_tokens, 0)
        self.assertEqual(direct.thinking_tokens, direct.tokens.index(FINAL))
        self.assertEqual(direct.stop, STOP_BUDGET)
        self.assertEqual(len(direct.tokens), 2)
        self.assertEqual(direct.tokens[-2], FINAL)
        self.assertIn(int(direct.tokens[-1]), direct.root_move_tokens)
        self.assertTrue(assess(direct).structural)

        budgeted = generate(self.model, [root], fixed_budget(3))[0]
        self.assertEqual(budgeted.thinking_tokens, 3)
        self.assertEqual(budgeted.thinking_tokens, budgeted.tokens.index(FINAL))
        self.assertEqual(budgeted.stop, STOP_BUDGET)
        self.assertEqual(len(budgeted.tokens), 5)
        self.assertNotIn(FINAL, budgeted.tokens[:3])
        self.assertEqual(budgeted.tokens[-2], FINAL)
        self.assertEqual(len(budgeted.tokens), budgeted.thinking_tokens + 2)
        assessment = assess(budgeted)
        self.assertTrue(assessment.structural)
        board = chess.Board(chess.STARTING_FEN)
        self.assertIn(chess.Move.from_uci(assessment.final_move), board.legal_moves)

    def test_autonomous_generation_records_its_stopping_length(self):
        root = RootPosition(chess.STARTING_FEN)
        capped = generate(self.model, [root], autonomous(0))[0]
        self.assertEqual(capped.stop, STOP_CAP)
        self.assertEqual(capped.thinking_tokens, 0)
        self.assertEqual(len(capped.tokens), 2)

    def test_generation_is_batched_consistently(self):
        roots = [RootPosition(chess.STARTING_FEN), RootPosition(SHORT_FEN)]
        together = generate(self.model, roots, fixed_budget(4), batch_size=8)
        alone = [generate(self.model, [root], fixed_budget(4))[0] for root in roots]
        self.assertEqual(
            [trace.tokens for trace in together],
            [trace.tokens for trace in alone],
        )
        self.assertEqual(
            [trace.thinking_tokens for trace in together],
            [trace.thinking_tokens for trace in alone],
        )

    def test_scripted_thought_generates_back_reset_and_final(self):
        torch.manual_seed(23)
        model = create_model(CONFIG)
        writer = memorized_trace(BLACK_TO_MOVE)
        self.memorize(model, BLACK_TO_MOVE, writer.tokens)
        scripted = tuple(int(token) for token in writer.tokens)

        trace = generate(model, [RootPosition(BLACK_TO_MOVE)], autonomous(32))[0]
        self.assertEqual(trace.tokens, scripted)
        self.assertEqual(trace.stop, STOP_FINAL)
        self.assertEqual(trace.thinking_tokens, 5)
        self.assertEqual(trace.thinking_tokens, trace.tokens.index(FINAL))
        self.assertEqual(len(trace.tokens), trace.thinking_tokens + 2)
        self.assertIn(BACK, trace.tokens)
        self.assertIn(RESET, trace.tokens)
        assessment = assess(trace)
        self.assertTrue(assessment.structural)
        self.assertEqual(assessment.max_depth, 1)
        self.assertEqual(assessment.root_width, 2)
        self.assertEqual(
            assessment.copy_first_move,
            sorted(move.uci() for move in chess.Board(BLACK_TO_MOVE).legal_moves)[0],
        )
        self.assertTrue(assessment.overturned)

        fixed = generate(model, [RootPosition(BLACK_TO_MOVE)], fixed_budget(4))[0]
        self.assertEqual(fixed.tokens[:4], scripted[:4])
        self.assertEqual(fixed.thinking_tokens, 4)
        self.assertEqual(fixed.stop, STOP_BUDGET)
        self.assertEqual(fixed.tokens[4], FINAL)
        self.assertIn(BACK, fixed.tokens)
        self.assertIn(RESET, fixed.tokens)
        self.assertTrue(assess(fixed).structural)

        # The model prefers FINAL after its fifth thought token. With a longer
        # fixed budget that token is masked, so the trace keeps thinking.
        ids = torch.from_numpy(encode_boards([BLACK_TO_MOVE]))
        prefix = torch.from_numpy(np.asarray([scripted[:6]], dtype=np.int16))
        preferred = int(model.trace_logits(ids, trace_ids=prefix)[0, -1].argmax())
        self.assertEqual(preferred, FINAL)
        long_budget = generate(model, [RootPosition(BLACK_TO_MOVE)], fixed_budget(7))[0]
        self.assertEqual(long_budget.thinking_tokens, 7)
        self.assertNotEqual(long_budget.tokens[5], FINAL)
        self.assertNotEqual(long_budget.tokens[6], FINAL)
        self.assertEqual(long_budget.tokens[7], FINAL)
        self.assertEqual(long_budget.stop, STOP_BUDGET)
        self.assertTrue(assess(long_budget).structural)

    def test_assessment_counts_evaluation_corrections_and_overturns(self):
        writer = TraceWriter(chess.STARTING_FEN)
        writer.move("e2e4")
        writer.evaluate(value=0.2)
        writer.evaluate(value=0.7)
        writer.back()
        writer.move("d2d4")
        writer.final("g1f3")
        trace = GeneratedTrace(
            chess.STARTING_FEN,
            1,
            writer.mirrored,
            tuple(int(token) for token in writer.tokens),
            5,
            STOP_BUDGET,
            "g1f3",
            (),
            (),
        )
        assessment = assess(trace)
        self.assertTrue(assessment.structural)
        self.assertEqual(assessment.evaluations, 2)
        self.assertEqual(assessment.evaluation_revisions, 1)
        self.assertEqual(assessment.max_depth, 1)
        self.assertEqual(assessment.max_width, 2)
        self.assertEqual(assessment.root_width, 2)
        self.assertEqual(assessment.copy_first_move, "e2e4")
        self.assertTrue(assessment.overturned)
        self.assertEqual(assessment.final_move, "g1f3")

    def test_assessment_reports_a_replay_failure(self):
        trace = GeneratedTrace(
            chess.STARTING_FEN,
            1,
            False,
            (FINAL, move_token("e7e5")),
            0,
            STOP_BUDGET,
            "e7e5",
            (),
            (),
        )
        assessment = assess(trace)
        self.assertFalse(assessment.structural)
        self.assertIn("final_illegal", assessment.error)
        self.assertIsNone(assessment.final_move)

    def test_move_probabilities_come_from_the_final_distribution(self):
        trace = generate(
            self.model, [RootPosition(chess.STARTING_FEN)], fixed_budget(0)
        )[0]
        probabilities = [
            math.exp(log_probability) for log_probability in trace.root_move_log_probs
        ]
        self.assertAlmostEqual(sum(probabilities), 1.0, places=5)
        self.assertAlmostEqual(
            move_log_probability(trace, trace.final_move),
            max(trace.root_move_log_probs),
            places=6,
        )
        with self.assertRaises(ValueError):
            move_log_probability(trace, "e2e5")

    def test_generation_uses_a_loaded_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequence.pth"
            save_checkpoint(path, self.model, CONFIG)
            loaded = load_model(path)
            root = RootPosition(chess.STARTING_FEN)
            expected = generate(self.model, [root], fixed_budget(2))[0]
            actual = generate(loaded, [root], fixed_budget(2))[0]
            self.assertEqual(actual.tokens, expected.tokens)
            self.assertEqual(actual.final_move, expected.final_move)

    def test_budget_grid_uses_the_fixed_fraction_points(self):
        self.assertEqual(p95_trace_length([8, 16, 100]), 100)
        self.assertEqual(trace_length_budget([9, 9, 9]), 16)
        self.assertEqual(budget_points([9, 9, 9]), (0, 4, 8, 12, 16, 24, 32))
        self.assertEqual(budget_points([8] * 20), (0, 2, 4, 6, 8, 12, 16))
        self.assertEqual(
            budget_grid({"fixed": [8] * 20, "adaptive": [100] * 20}),
            (0, 2, 4, 6, 8, 12, 16, 26, 52, 78, 104, 156, 208),
        )
        with self.assertRaises(ValueError):
            budget_grid({"empty": []})

    def test_token_counts_strip_only_board_content(self):
        board = chess.Board(chess.STARTING_FEN)
        legal = sorted(move.uci() for move in board.legal_moves)
        tool = TraceWriter(chess.STARTING_FEN)
        tool.board()
        tool.move(legal[0])
        tool.evaluate(value=0.25)
        tool.back()
        tool.final(legal[1])
        plain = TraceWriter(chess.STARTING_FEN)
        plain.move(legal[0])
        plain.evaluate(value=0.25)
        plain.back()
        plain.final(legal[1])
        self.assertEqual(
            no_tool_token_count(tool.tokens, chess.STARTING_FEN),
            len(plain.tokens),
        )
        self.assertEqual(
            model_token_count(tool.tokens, chess.STARTING_FEN),
            len(plain.tokens) + 1,
        )
        # The rule-free scan and the replay-checked count agree on the real
        # trace, including PAD padding.
        self.assertEqual(no_tool_length(tool.tokens), len(plain.tokens))
        padded = np.concatenate([tool.tokens, [PAD, PAD, PAD]])
        self.assertEqual(no_tool_length(padded), len(plain.tokens))


if __name__ == "__main__":
    unittest.main()
