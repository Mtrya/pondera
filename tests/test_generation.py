import unittest

import chess
import torch

from pondera.chess.merge import MergeDictionary
from pondera.chess.vocabulary import (
    BOARD,
    CHESS_VOCAB_SIZE,
    DEPTH,
    END,
    FINAL,
    MOVE,
    UCI_MOVE_TO_IDX,
    VALUE,
)
from pondera.data.trace import TraceWriter, replay
from pondera.inference.policy import pick_move
from pondera.inference.trace import (
    STOP_BUDGET,
    STOP_CAP,
    STOP_FINAL,
    GeneratedTrace,
    RootPosition,
    _choose,
    _GenerationRow,
    assess,
    autonomous,
    fixed_budget,
    generate,
    legal_tokens,
    move_log_probability,
)
from pondera.models.policy import create_model

CONFIG = {
    "num_blocks": 1,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_heads": 2,
    "dropout": 0.0,
}
BLACK_TO_MOVE = chess.Board()
BLACK_TO_MOVE.push_uci("e2e4")
BLACK_TO_MOVE = BLACK_TO_MOVE.fen(en_passant="fen")


def move_token(uci):
    return MOVE + UCI_MOVE_TO_IDX[uci]


class GenerationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.model = create_model(CONFIG).eval()

    def test_board_and_its_atoms_are_model_generated(self):
        row = _GenerationRow(RootPosition(chess.STARTING_FEN), fixed_budget(100))
        row.writer.depth()
        row.writer.move("e2e4")
        logits = torch.zeros((1, CHESS_VOCAB_SIZE))
        logits[0, BOARD] = 1.0
        choices, _ = _choose([row], logits, fixed_budget(100))
        self.assertEqual(int(choices[0]), BOARD)
        row.apply(int(choices[0]), False)
        atom_count = 0
        while row.writer.completing_board:
            choices, _ = _choose(
                [row], torch.zeros((1, CHESS_VOCAB_SIZE)), fixed_budget(100)
            )
            allowed = legal_tokens(row.writer, False, row.legal_cache)
            self.assertEqual(len(allowed), 1)
            self.assertEqual(int(choices[0]), allowed[0])
            row.apply(int(choices[0]), False)
            atom_count += 1
        self.assertEqual(atom_count, 40)
        row.writer.final("d2d4")
        row.writer.final_evaluate(value=0.0)
        row.writer.end()
        program = replay(row.writer.tokens, chess.STARTING_FEN)
        board_step = next(step for step in program.steps if step.operation == "board")
        after = chess.Board(chess.STARTING_FEN)
        after.push_uci("e2e4")
        self.assertEqual(board_step.position, after.fen(en_passant="fen"))
        self.assertEqual(
            board_step.observed.split()[:4], program.steps[1].position.split()[:4]
        )

    def test_stopping_protocols_close_a_rule_valid_trace(self):
        root = RootPosition(chess.STARTING_FEN)
        for protocol, stop in (
            (fixed_budget(0), STOP_BUDGET),
            (fixed_budget(3), STOP_BUDGET),
            (autonomous(0), STOP_CAP),
        ):
            with self.subTest(protocol=protocol):
                trace = generate(self.model, [root], protocol)[0]
                self.assertEqual(trace.stop, stop)
                self.assertTrue(assess(trace).structural)
                self.assertIn(FINAL, trace.tokens)

    def test_generation_is_batched_consistently(self):
        roots = [RootPosition(chess.STARTING_FEN), RootPosition(BLACK_TO_MOVE)]
        together = generate(self.model, roots, fixed_budget(4), batch_size=2)
        alone = [generate(self.model, [root], fixed_budget(4))[0] for root in roots]
        self.assertEqual(
            [trace.tokens for trace in together], [trace.tokens for trace in alone]
        )

    def test_merged_generation_validates_expansions_and_selects_legal_moves(self):
        # The higher-scoring expansion contains an illegal e2e5 decision.
        entries = (
            (FINAL, move_token("e2e5"), VALUE, END),
            (FINAL, move_token("e2e4"), VALUE, END),
        )
        model = create_model({**CONFIG, "merge_entries": entries}).eval()
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
            model.head.bias[CHESS_VOCAB_SIZE] = 20
            model.head.bias[CHESS_VOCAB_SIZE + 1] = 10
        roots = [RootPosition(chess.STARTING_FEN), RootPosition(BLACK_TO_MOVE)]
        for protocol in (fixed_budget(0), autonomous(0), autonomous(5)):
            traces = generate(model, roots, protocol)
            self.assertEqual([trace.final_move for trace in traces], ["e2e4", "e7e5"])
            for trace in traces:
                self.assertTrue(assess(trace).structural)
                self.assertEqual(trace.tokens, entries[1])
                self.assertAlmostEqual(
                    sum(torch.tensor(trace.root_move_log_probs).exp().tolist()), 1.0
                )
        self.assertEqual(pick_move(model, chess.Board()), "e2e4")

    def test_composites_cannot_cross_the_thinking_budget_or_trail_end(self):
        dictionary = MergeDictionary(
            (
                (DEPTH, move_token("e2e4")),
                (DEPTH, FINAL, move_token("e2e4"), VALUE, END),
                (FINAL, move_token("e2e4"), VALUE, END, END),
            )
        )
        row = _GenerationRow(RootPosition(chess.STARTING_FEN), fixed_budget(1))
        logits = torch.zeros((1, CHESS_VOCAB_SIZE + len(dictionary)))
        logits[0, CHESS_VOCAB_SIZE] = 30
        logits[0, CHESS_VOCAB_SIZE + 1] = 20
        logits[0, CHESS_VOCAB_SIZE + 2] = 40
        choices, _ = _choose([row], logits, fixed_budget(1), dictionary)
        self.assertEqual(int(choices[0]), CHESS_VOCAB_SIZE + 1)
        self.assertEqual(len(row.writer.tokens), 0)

    def test_generated_composites_feed_the_next_model_step(self):
        entry = (DEPTH, move_token("e2e4"))
        model = create_model({**CONFIG, "merge_entries": (entry,)}).eval()
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
            model.head.bias[CHESS_VOCAB_SIZE] = 10
        trace = generate(model, [RootPosition(chess.STARTING_FEN)], fixed_budget(2))[0]
        self.assertEqual(trace.tokens[:2], entry)
        self.assertEqual(trace.thinking_tokens, 2)
        self.assertTrue(assess(trace).structural)

    def test_assess_replays_with_the_declared_teacher_width(self):
        legal = sorted(move.uci() for move in chess.Board().legal_moves)
        writer = TraceWriter(chess.STARTING_FEN)
        writer.depth()
        writer.move(legal[0])
        writer.evaluate(value=0.1)
        writer.reset()
        writer.move(legal[1])
        writer.refuted()
        writer.reset()
        writer.final(legal[2])
        writer.final_evaluate(value=0.1)
        writer.end()
        tokens = tuple(int(token) for token in writer.tokens)

        def trace(width):
            return GeneratedTrace(
                chess.STARTING_FEN,
                1,
                False,
                tokens,
                tokens.index(FINAL),
                STOP_FINAL,
                legal[2],
                (),
                (),
                width,
            )

        self.assertTrue(assess(trace(1)).structural)
        self.assertFalse(assess(trace(3)).structural)

    def test_final_move_probabilities_are_normalized_over_legal_moves(self):
        trace = generate(
            self.model, [RootPosition(chess.STARTING_FEN)], fixed_budget(0)
        )[0]
        self.assertAlmostEqual(
            sum(torch.tensor(trace.root_move_log_probs).exp().tolist()),
            1.0,
            places=6,
        )
        for move in chess.Board().legal_moves:
            self.assertEqual(
                move_log_probability(trace, move.uci()),
                trace.root_move_log_probs[
                    trace.root_move_tokens.index(move_token(move.uci()))
                ],
            )


if __name__ == "__main__":
    unittest.main()
