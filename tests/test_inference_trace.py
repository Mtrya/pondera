"""Expanded-answer contracts and generic generation on real model variants."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from pondera.chess import _rules as rules
from pondera.chess.codec import encode_board
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
    RESET,
    UCI_MOVE_TO_IDX,
    VALUE,
)
from pondera.inference.trace import (
    Predictor,
    _DecodeState,
    load_checkpoint,
    parse_answer,
)
from pondera.inference.vanilla_uci import pick_with_reasoning
from pondera.models import PonderaModel


def move(uci):
    return MOVE + UCI_MOVE_TO_IDX[uci]


class AnswerTests(unittest.TestCase):
    def test_equivalent_segmentations_and_opaque_reasoning(self):
        a, b, c = move("e2e4"), VALUE + 10, move("e7e5")
        dictionary = MergeDictionary([(a, b), (b, c)])
        forms = ([a, CHESS_VOCAB_SIZE + 1], [CHESS_VOCAB_SIZE, c], [a, b, c])
        for turn, decision, reply in (("w", "e2e4", "e7e5"), ("b", "e7e5", "e2e4")):
            fen = rules.STARTING_FEN.replace(" w ", f" {turn} ")
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
            board = rules.Board(fen)
            board.push_uci(decision)
            self.assertIn(rules.Move.from_uci(reply), board.legal_moves)

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
                answer = parse_answer(tokens, MergeDictionary(), rules.STARTING_FEN)
                self.assertFalse(answer.valid)
                self.assertIn(error, answer.errors)
        partial = parse_answer(cases[1][0], MergeDictionary(), rules.STARTING_FEN)
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
                    tokens, MergeDictionary(), rules.STARTING_FEN, repetition
                ).valid,
                legal,
            )
        invalid = parse_answer(
            tokens[:-1] + [move("e2e4"), END], MergeDictionary(), rules.STARTING_FEN, 3
        )
        self.assertFalse(invalid.valid)
        loss = parse_answer(
            [FINAL, move("e2e4"), VALUE, MATE_LOSS + 5, END],
            MergeDictionary(),
            rules.STARTING_FEN,
        )
        self.assertTrue(loss.valid)
        self.assertEqual(loss.mate_distance, -5)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_cached_predictions_equal_full_context_after_final_insertion(self):
        torch.manual_seed(640)
        prefix = encode_board(rules.STARTING_FEN).tolist()
        tokens = [BACK, move("e2e4"), BACK, FINAL, VALUE, END]
        model = PonderaModel(
            segments=[
                {"blocks": [{"attention": "kda", "group_end": True}]},
                {
                    "repeat": 2,
                    "blocks": ["kda", {"attention": "softmax", "group_end": True}],
                },
                {
                    "blocks": [
                        {"attention": "softmax", "value_source": "token_embedding"}
                    ]
                },
            ],
            block_attn_res=True,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            num_kv_heads=1,
            vocab_size=CHESS_VOCAB_SIZE,
        ).eval()
        predictor = Predictor(model, MergeDictionary(), {})
        state = _DecodeState()
        with torch.inference_mode():
            # Inserting FINAL can leave two tokens to prefill together.
            for length in (0, 1, 2, 4, 5, 6):
                context = torch.tensor(prefix + tokens[:length])
                expected = model(context, [0, len(context)])[-1]
                actual = predictor._next_logits(prefix, tokens[:length], state)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
                self.assertEqual(actual.argmax().item(), expected.argmax().item())

    def test_chess_path_reads_are_causal_across_packing_and_cached_decode(self):
        torch.manual_seed(640)
        dictionary = MergeDictionary([(BACK, VALUE + 10)])
        prefix = encode_board(rules.STARTING_FEN).tolist()
        e4, e5, back = move("e2e4"), move("e7e5"), CHESS_VOCAB_SIZE
        trace = [e4, e5, back, move("c7c5"), RESET, FINAL, e4]
        document = torch.tensor(prefix + trace)
        packed = document.repeat(2)
        first, length = len(prefix), len(document)
        config = dict(
            segments=[
                {"blocks": [{"attention": "kda", "group_end": True}]},
                {
                    "repeat": 2,
                    "blocks": [
                        "kda",
                        {"attention": "softmax", "navigation": True, "group_end": True},
                    ],
                },
                {
                    "blocks": [
                        {"attention": "softmax", "value_source": "token_embedding"}
                    ]
                },
            ],
            block_attn_res=True,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            num_kv_heads=1,
            vocab_size=CHESS_VOCAB_SIZE + 1,
        )
        with torch.inference_mode():
            model = PonderaModel(**config)
            for read in model.depth_reads:
                read.query.normal_(std=0.15)
            model.navigation["2"].query.normal_(std=0.25)
            with TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint.pth"
                torch.save(
                    dict(
                        arch=model.architecture,
                        model_config=config,
                        model_state_dict=model.state_dict(),
                        tokenizer_identity={"merge_entries": dictionary.entries},
                    ),
                    path,
                )
                predictor = load_checkpoint(path)
            model = predictor.model
            receivers, senders = model.prepare_navigation(
                packed.numpy(), [0, length, 2 * length]
            )
            # BACK returns to the first MOVE. The second MOVE was written last.
            self.assertEqual(
                receivers.tolist(),
                [first + 2, first + 4, length + first + 2, length + first + 4],
            )
            self.assertEqual(
                senders[[0, 2]].tolist(),
                [[first, first + 1], [length + first, length + first + 1]],
            )
            full = model(document, [0, length])
            torch.testing.assert_close(
                model(packed, [0, length, 2 * length])[length:], full
            )
            cache = _DecodeState()
            for size in (0, 1, 2, 3, 4, 5, 7):
                actual = predictor._next_logits(prefix, trace[:size], cache)
                torch.testing.assert_close(
                    actual, full[first + size - 1], atol=1e-5, rtol=1e-5
                )

    def model(self):
        model = PonderaModel(
            segments=[{"blocks": ["softmax"]}],
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
        predictor = Predictor(self.model(), MergeDictionary(), {})
        result = predictor.generate(
            rules.STARTING_FEN, reasoning_tokens=3, answer_tokens=4
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
            rules.STARTING_FEN, reasoning_tokens=0, answer_tokens=2
        )
        self.assertEqual(zero.tokens, [FINAL, 0, 0])
        self.assertEqual(zero.generated_tokens, 2)

    def test_natural_end_is_never_repaired(self):
        model = self.model()
        with torch.no_grad():
            model.token_embedding.weight.fill_(1)
            model.head.weight[END].fill_(1)
        predictor = Predictor(model, MergeDictionary(), {})
        result = predictor.generate(
            rules.STARTING_FEN, reasoning_tokens=3, answer_tokens=4
        )
        self.assertEqual(result.tokens, [END])
        self.assertEqual(result.stop, "end_before_final")
        self.assertFalse(result.forced_final)
        self.assertFalse(result.answer.valid)
        result = predictor.generate(rules.STARTING_FEN, answer_tokens=4)
        self.assertEqual(result.tokens, [FINAL, END])
        self.assertFalse(result.answer.valid)

    def test_uci_move_can_depend_on_generated_reasoning(self):
        model = self.model()
        with torch.no_grad():
            model.token_embedding.weight[:, 3] = 1
            model.token_embedding.weight[FINAL, 3] = 0
            model.token_embedding.weight[FINAL, 0] = 1
            model.token_embedding.weight[BACK, 1] = 1
            model.blocks[0].attention.v.weight[1, 1] = 1
            model.blocks[0].attention.o.weight[1, 1] = 1
            model.head.weight[BACK, 3] = 1
            model.head.weight[move("d2d4"), 0] = 0.01
            model.head.weight[move("e2e4"), 1] = 1

        # Uniform causal attention gives e4 its only signal from the emitted BACK.
        predictor = Predictor(model, MergeDictionary(), {})
        board = rules.Board()
        self.assertEqual(
            pick_with_reasoning(predictor, board, "cpu", reasoning_tokens=0), "d2d4"
        )
        self.assertEqual(
            pick_with_reasoning(predictor, board, "cpu", reasoning_tokens=1), "e2e4"
        )


if __name__ == "__main__":
    unittest.main()
