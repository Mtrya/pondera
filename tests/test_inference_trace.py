"""Expanded-answer contracts and generic generation on both real model variants."""

import unittest

import chess
import torch

from pondera.chess.merge import MergeDictionary
from pondera.chess.vocabulary import (
    BACK,
    BOARD,
    CHESS_VOCAB_SIZE,
    END,
    FINAL,
    MATE_LOSS,
    MATE_WIN,
    MOVE,
    UCI_MOVE_TO_IDX,
    VALUE,
)
from pondera.inference.trace import Predictor, parse_answer
from pondera.models.linear import LinearVanillaModel
from pondera.models.vanilla import VanillaModel


def move(uci):
    return MOVE + UCI_MOVE_TO_IDX[uci]


class AnswerTests(unittest.TestCase):
    def test_equivalent_segmentations_and_opaque_reasoning(self):
        a, b, c = move("e2e4"), VALUE + 10, move("e7e5")
        dictionary = MergeDictionary([(a, b), (b, c)])
        forms = ([a, CHESS_VOCAB_SIZE + 1], [CHESS_VOCAB_SIZE, c], [a, b, c])
        for turn, decision, reply in (("w", "e2e4", "e7e5"), ("b", "e7e5", "e2e4")):
            fen = chess.STARTING_FEN.replace(" w ", f" {turn} ")
            parsed = [
                parse_answer(
                    [BOARD, BACK, VALUE + 4, FINAL, *form, END], dictionary, fen
                )
                for form in forms
            ]
            self.assertTrue(all(answer.valid for answer in parsed))
            self.assertEqual(parsed[0], parsed[1])
            self.assertEqual(parsed[1], parsed[2])
            self.assertEqual(parsed[0].decision, decision)
            self.assertEqual(parsed[0].pv, [reply])
            board = chess.Board(fen)
            board.push_uci(decision)
            self.assertIn(chess.Move.from_uci(reply), board.legal_moves)

    def test_malformed_or_illegal_answers_fail_without_discarding_move_diagnostics(
        self,
    ):
        cases = [
            ([move("e2e4"), VALUE + 10, END], "missing_final"),
            ([FINAL, move("e2e4"), VALUE + 10], "missing_end"),
            ([FINAL, move("e2e4"), END], "missing_value"),
            ([FINAL, move("e2e5"), VALUE + 10, END], "illegal_decision"),
            ([FINAL, move("e2e4"), VALUE + 10, move("e7e4"), END], "illegal_pv"),
            (
                [FINAL, move("e2e4"), VALUE + 10, MATE_WIN + 1, END],
                "mate_value_mismatch",
            ),
            ([FINAL, move("e2e4"), VALUE + 10, FINAL, END], "multiple_final"),
            ([FINAL, move("e2e4"), VALUE + 10, END, VALUE + 10], "trailing_after_end"),
        ]
        for tokens, error in cases:
            with self.subTest(error=error):
                answer = parse_answer(tokens, MergeDictionary(), chess.STARTING_FEN)
                self.assertFalse(answer.valid)
                self.assertIn(error, answer.errors)
        partial = parse_answer(cases[1][0], MergeDictionary(), chess.STARTING_FEN)
        self.assertTrue(partial.move_legal)
        self.assertFalse(partial.format_valid)

    def test_pv_cannot_continue_after_checkmate(self):
        fen = "7k/5K2/6Q1/8/8/8/8/8 w - - 0 1"
        answer = parse_answer(
            [FINAL, move("g6g7"), VALUE + 19, MATE_WIN + 1, move("h8h7"), END],
            MergeDictionary(),
            fen,
        )
        self.assertTrue(answer.format_valid)
        self.assertTrue(answer.move_legal)
        self.assertFalse(answer.pv_legal)
        self.assertFalse(answer.valid)

    def test_draw_claim_requires_rights_and_no_pv(self):
        tokens = [FINAL, move("<claim_draw>"), VALUE + 10, END]
        for repetition, legal in ((1, False), (3, True)):
            self.assertEqual(
                parse_answer(
                    tokens, MergeDictionary(), chess.STARTING_FEN, repetition
                ).valid,
                legal,
            )
        invalid = parse_answer(
            tokens[:-1] + [move("e2e4"), END], MergeDictionary(), chess.STARTING_FEN, 3
        )
        self.assertFalse(invalid.valid)
        loss = parse_answer(
            [FINAL, move("e2e4"), VALUE, MATE_LOSS + 5, END],
            MergeDictionary(),
            chess.STARTING_FEN,
        )
        self.assertTrue(loss.valid)
        self.assertEqual(loss.mate_distance, -5)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def model(self, model_class):
        model = model_class(
            num_blocks=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            num_kv_heads=1,
            vocab_size=CHESS_VOCAB_SIZE,
        )
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim == 2:
                    parameter.zero_()
        return model

    def test_unconstrained_generation_budgets_count_model_steps(self):
        # Zero logits choose token 0, a board atom, at every step. An answer
        # mask or repair would change these analytically fixed outputs.
        for model_class in (VanillaModel, LinearVanillaModel):
            with self.subTest(architecture=model_class.architecture):
                predictor = Predictor(self.model(model_class), MergeDictionary(), {})
                result = predictor.generate(
                    chess.STARTING_FEN, reasoning_tokens=3, answer_tokens=4
                )
                self.assertEqual(result.tokens, [0, 0, 0, FINAL, 0, 0, 0, 0])
                self.assertEqual(
                    (
                        result.reasoning_tokens,
                        result.answer_tokens,
                        result.generated_tokens,
                    ),
                    (3, 4, 7),
                )
                self.assertTrue(result.forced_final)
                self.assertFalse(result.answer.valid)
                zero = predictor.generate(
                    chess.STARTING_FEN, reasoning_tokens=0, answer_tokens=2
                )
                self.assertEqual(zero.tokens, [FINAL, 0, 0])
                self.assertEqual(zero.generated_tokens, 2)

    def test_natural_end_is_never_repaired(self):
        for model_class in (VanillaModel, LinearVanillaModel):
            model = self.model(model_class)
            with torch.no_grad():
                model.token_embedding.weight.fill_(1)
                model.head.weight[END].fill_(1)
            predictor = Predictor(model, MergeDictionary(), {})
            result = predictor.generate(
                chess.STARTING_FEN, reasoning_tokens=3, answer_tokens=4
            )
            self.assertEqual(result.tokens, [END])
            self.assertEqual(result.stop, "end_before_final")
            self.assertFalse(result.forced_final)
            self.assertFalse(result.answer.valid)
            result = predictor.generate(chess.STARTING_FEN, answer_tokens=4)
            self.assertEqual(result.tokens, [FINAL, END])
            self.assertFalse(result.answer.valid)


if __name__ == "__main__":
    unittest.main()
