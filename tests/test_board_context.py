"""Independent chess oracle and causal execution of board conditioning."""

import random
import unittest
from itertools import product

import chess
import torch

from pondera.chess.codec import encode_board
from pondera.chess.vocabulary import BOARD_TOKENS, CHESS_VOCAB_SIZE, FINAL
from pondera.models.board_context import (
    BoardContextModel,
    CompiledBoardFeatures,
    SymbolicBoardModel,
)
from pondera.models.delta import GatedDeltaModel, KimiDeltaModel
from pondera.models.vanilla import VanillaModel
from pondera.training.muon import build_optimizers


def board_ids(board):
    return torch.tensor(encode_board(board.fen()), dtype=torch.long)


class CompiledBoardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.features = CompiledBoardFeatures()

    def test_attacks_and_counts_match_python_chess_including_partial_boards(self):
        rng = random.Random(640)
        board = chess.Board()
        positions = [board.copy()]
        for ply in range(160):
            if board.is_game_over():
                board = chess.Board()
            board.push(rng.choice(list(board.legal_moves)))
            if ply % 8 == 0:
                positions.append(board.copy())
        # Pins, friendly ray endpoints and blocked sliders remain geometric attacks.
        positions.append(chess.Board("4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1"))
        for position in positions:
            ids = board_ids(position)
            for length in (2, len(ids) // 2, len(ids)):
                prefix = ids[:length]
                partial = chess.Board(None)
                for token in prefix.tolist():
                    if token < BOARD_TOKENS:
                        square, piece = divmod(token, 12)
                        partial.set_piece_at(
                            square, chess.Piece.from_symbol("PNBRQKpnbrqk"[piece])
                        )
                expected = [
                    len(partial.pieces(piece, color))
                    for color in (chess.WHITE, chess.BLACK)
                    for piece in range(chess.PAWN, chess.KING + 1)
                ] + [
                    len(partial.attackers(color, square))
                    for color in (chess.WHITE, chess.BLACK)
                    for square in chess.SQUARES
                ]
                actual = self.features(self.features.embedding(prefix).sum(0))
                torch.testing.assert_close(
                    actual, torch.tensor(expected).float(), rtol=0, atol=0
                )


class BoardContextTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(640)
        board = chess.Board()
        self.first = torch.cat((board_ids(board), torch.tensor([FINAL, 19, 28])))
        board.push_uci("e2e4")
        self.second = torch.cat((board_ids(board), torch.tensor([FINAL, 36])))
        self.config = dict(
            num_blocks=2,
            hidden_size=32,
            intermediate_size=48,
            num_heads=4,
            num_kv_heads=2,
            vocab_size=CHESS_VOCAB_SIZE,
        )

    def test_packed_causality_and_cached_decoding_through_prefix_boundary(self):
        for cls, mixer in product(
            (BoardContextModel, SymbolicBoardModel), ("softmax", "gdn", "kda")
        ):
            with (
                self.subTest(architecture=cls.architecture, mixer=mixer),
                torch.no_grad(),
            ):
                model = cls(
                    **self.config,
                    mixer=mixer,
                    softmax_every=0 if mixer == "softmax" else 2,
                ).eval()
                ids = torch.cat((self.first, self.second))
                cu = [0, len(self.first), len(ids)]
                actual = model(ids, cu)
                separate = torch.cat(
                    (
                        model(self.first, [0, len(self.first)]),
                        model(self.second, [0, len(self.second)]),
                    )
                )
                torch.testing.assert_close(actual, separate, rtol=1e-5, atol=1e-6)
                changed = ids.clone()
                changed[7:] = (changed[7:] + 17) % CHESS_VOCAB_SIZE
                torch.testing.assert_close(
                    model(changed, cu)[:7], actual[:7], rtol=0, atol=0
                )
                prefix, cache = model.forward_cached(
                    self.first[:5], torch.tensor([0, 5])
                )
                parts = [prefix]
                for index in range(5, len(self.first)):
                    out, cache = model.forward_cached(
                        self.first[index : index + 1],
                        torch.tensor([0, index + 1]),
                        cache,
                    )
                    parts.append(out)
                cached, expected = torch.cat(parts), actual[: len(self.first)]
                # Incremental convolution/GEMM changes fp32 reduction order.
                torch.testing.assert_close(cached, expected, rtol=1e-5, atol=1e-5)
                self.assertLess(
                    float((cached - expected).norm() / expected.norm()), 2e-6
                )

    def test_recomputation_gradients_and_frozen_circuit_during_training(self):
        for cls, mixer in product(
            (BoardContextModel, SymbolicBoardModel), ("softmax", "gdn", "kda")
        ):
            with self.subTest(architecture=cls.architecture, mixer=mixer):
                options = dict(
                    mixer=mixer, softmax_every=0 if mixer == "softmax" else 2
                )
                model, recomputed = (
                    cls(**self.config, **options).train(),
                    cls(**self.config, **options).train(),
                )
                recomputed.load_state_dict(model.state_dict())
                recomputed.activation_checkpointing = True
                targets = self.first.roll(-1)
                for network in (model, recomputed):
                    torch.nn.functional.cross_entropy(
                        network(self.first, [0, len(self.first)]), targets
                    ).backward()
                for actual, expected in zip(
                    recomputed.parameters(), model.parameters(), strict=True
                ):
                    if expected.requires_grad:
                        torch.testing.assert_close(
                            actual.grad, expected.grad, rtol=0, atol=0
                        )
                    else:
                        self.assertIsNone(actual.grad)
                frozen = {
                    name: p.detach().clone()
                    for name, p in model.named_parameters()
                    if not p.requires_grad
                }
                embedding_before = model.token_embedding.weight.detach().clone()
                for optimizer in build_optimizers(model):
                    optimizer.step()
                self.assertFalse(
                    torch.equal(embedding_before, model.token_embedding.weight)
                )
                for name, p in model.named_parameters():
                    if name in frozen:
                        torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)

    def test_zero_modulation_preserves_backbone_initialization_outputs_and_gradients(
        self,
    ):
        for cls, (mixer, backbone_cls) in product(
            (BoardContextModel, SymbolicBoardModel),
            (
                ("softmax", VanillaModel),
                ("gdn", GatedDeltaModel),
                ("kda", KimiDeltaModel),
            ),
        ):
            with self.subTest(architecture=cls.architecture, mixer=mixer):
                options = {} if mixer == "softmax" else {"softmax_every": 2}
                torch.manual_seed(640)
                backbone = backbone_cls(**self.config, **options)
                torch.manual_seed(640)
                conditioned = cls(**self.config, mixer=mixer, **options)
                parameters = {
                    name.replace(".attention.attention.", ".attention."): p
                    for name, p in conditioned.named_parameters()
                    if not name.startswith("features.") and ".context_" not in name
                }
                for name, p in backbone.named_parameters():
                    torch.testing.assert_close(parameters[name], p, rtol=0, atol=0)
                with torch.no_grad():
                    for block in conditioned.blocks:
                        block.attention.context_up.weight.zero_()
                expected = backbone(self.first, [0, len(self.first)])
                actual = conditioned(self.first, [0, len(self.first)])
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for logits in (actual, expected):
                    torch.nn.functional.cross_entropy(
                        logits, self.first.roll(-1)
                    ).backward()
                for name, p in backbone.named_parameters():
                    torch.testing.assert_close(
                        parameters[name].grad, p.grad, rtol=0, atol=0
                    )

    def test_summary_ignores_later_trace_boards(self):
        for cls in (BoardContextModel, SymbolicBoardModel):
            with self.subTest(architecture=cls.architecture):
                model = cls(**self.config)
                # A short board makes subsequent board atoms fall inside the 40-token gather.
                prefix = board_ids(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"))
                ids = torch.cat((prefix, torch.tensor([FINAL, 0, 12, 24])))
                context, indices = model._prepare_context(ids, [0, len(ids)])
                observed = context[indices]
                torch.testing.assert_close(
                    observed[len(prefix) :],
                    observed[len(prefix) - 1].expand_as(observed[len(prefix) :]),
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
