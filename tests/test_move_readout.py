import unittest

import chess
import torch
import torch.nn.functional as F

from pondera.chess.board_tokens import encode_boards
from pondera.chess.mapping import (
    CLAIM_DRAW_FROM,
    IDX_TO_UCI_MOVE,
    MOVE_FROM,
    MOVE_PROMO,
    MOVE_TO,
    PROMO_TO_IDX,
    REFLECTED_SQUARE,
    UCI_MOVE_TO_IDX,
)
from pondera.models.checkpoint import load_model, save_checkpoint
from pondera.models.factory import create_model
from pondera.training.metrics import PolicyMetrics, factored_nll, factored_policy_loss

CONFIG = {
    "architecture": "base",
    "num_blocks": 2,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_heads": 2,
}

FENS = [
    chess.STARTING_FEN,
    "r3k2r/8/8/3pP3/8/8/8/R3K2R b KQkq d6 0 1",
    "7k/8/8/8/8/8/p7/7K b - - 0 80",
]


def factor_tables(device="cpu"):
    return tuple(
        torch.tensor(table, device=device)
        for table in (MOVE_FROM, MOVE_TO, MOVE_PROMO)
    )


class MoveReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_factor_tables_match_python_chess(self):
        claim = UCI_MOVE_TO_IDX["<claim_draw>"]
        for index, uci in IDX_TO_UCI_MOVE.items():
            if index == claim:
                self.assertEqual(
                    (MOVE_FROM[index], MOVE_TO[index], MOVE_PROMO[index]),
                    (CLAIM_DRAW_FROM, -1, -1),
                )
                continue
            move = chess.Move.from_uci(uci)
            self.assertEqual(MOVE_FROM[index], move.from_square)
            self.assertEqual(MOVE_TO[index], move.to_square)
            expected_promo = (
                PROMO_TO_IDX[chess.piece_symbol(move.promotion)]
                if move.promotion
                else PROMO_TO_IDX[""]
            )
            self.assertEqual(MOVE_PROMO[index], expected_promo)
        for square in range(64):
            self.assertEqual(REFLECTED_SQUARE[square], chess.square_mirror(square))

    def test_factored_materialization_is_normalized_factorization(self):
        model = create_model({**CONFIG, "readout": "factored"}).eval()
        ids = torch.from_numpy(encode_boards(FENS))
        factors, _ = model(ids=ids, return_factors=True)
        joint = model.move_log_probs(factors)
        self.assertEqual(joint.shape, (len(FENS), len(IDX_TO_UCI_MOVE)))
        from_logits, to_logits, promo_logits = factors
        lf = F.log_softmax(from_logits.float(), -1)
        lt = F.log_softmax(to_logits.float(), -1)
        lp = F.log_softmax(promo_logits.float(), -1)
        # Spot-check moves across files/promotions: joint log-prob is the sum
        # of the three factor log-probs (claim_draw carries only the from slot).
        claim = UCI_MOVE_TO_IDX["<claim_draw>"]
        for uci in ("e2e4", "a7a8q", "h2h1n", "d1h5", "<claim_draw>"):
            index = UCI_MOVE_TO_IDX[uci]
            expected = lf[:, MOVE_FROM[index]]
            if index != claim:
                expected = (
                    expected + lt[:, MOVE_TO[index]] + lp[:, MOVE_PROMO[index]]
                )
            torch.testing.assert_close(joint[:, index], expected)
        # Sub-normalization: total mass on the valid vocabulary is at most one.
        self.assertTrue((joint.exp().sum(-1) <= 1 + 1e-5).all())

    def test_factored_reflection_equivariance(self):
        model = create_model(
            {
                **CONFIG,
                "readout": "factored",
                "orientation": "side_to_move",
                "layout": "pieces",
            }
        ).eval()
        torch.manual_seed(0)
        from_logits = torch.randn(3, CLAIM_DRAW_FROM + 1)
        to_logits = torch.randn(3, 64)
        promo_logits = torch.randn(3, 5)
        permuted_joint = model.move_log_probs(
            (from_logits, to_logits, promo_logits)
        )[:, model.reflected_actions]
        reflected_joint = model.move_log_probs(
            (
                from_logits[:, model.reflected_from],
                to_logits[:, model.reflected_to],
                promo_logits,
            )
        )
        torch.testing.assert_close(permuted_joint, reflected_joint)

    def test_factored_loss_and_gradients(self):
        model = create_model(
            {
                **CONFIG,
                "readout": "factored",
                "orientation": "side_to_move",
                "layout": "pieces",
            }
        )
        ids = torch.from_numpy(encode_boards(FENS))
        boards = [chess.Board(fen) for fen in FENS]
        targets = torch.tensor(
            [UCI_MOVE_TO_IDX[next(iter(board.legal_moves)).uci()] for board in boards]
        )
        factors, _ = model(ids=ids, return_factors=True)
        loss = factored_policy_loss(factors, targets, factor_tables())
        # Independent reference: plain cross-entropy on each factor head.
        move_from, move_to, move_promo = factor_tables()
        reference = (
            F.cross_entropy(from_logits := factors[0].float(), move_from[targets])
            + F.cross_entropy(factors[1].float(), move_to[targets])
            + F.cross_entropy(factors[2].float(), move_promo[targets])
        )
        torch.testing.assert_close(loss, reference)
        loss.backward()
        for head in (model.from_proj, model.to_proj, model.promo_proj):
            self.assertGreater(float(head.weight.grad.abs().sum()), 0)
        # Per-position NLL equals the sum of per-factor cross-entropies.
        nll = factored_nll(factors, targets, factor_tables())
        per_position = (
            F.cross_entropy(from_logits, move_from[targets], reduction="none")
            + F.cross_entropy(factors[1].float(), move_to[targets], reduction="none")
            + F.cross_entropy(
                factors[2].float(), move_promo[targets], reduction="none"
            )
        )
        torch.testing.assert_close(nll, per_position)

    def test_factored_metrics_carry_teacher_nll(self):
        model = create_model({**CONFIG, "readout": "factored"}).eval()
        ids = torch.from_numpy(encode_boards(FENS))
        factors, _ = model(ids=ids, return_factors=True)
        joint = model.move_log_probs(factors)
        boards = [chess.Board(fen) for fen in FENS]
        targets = torch.tensor(
            [UCI_MOVE_TO_IDX[next(iter(board.legal_moves)).uci()] for board in boards]
        )
        legal = torch.tensor(
            [
                [UCI_MOVE_TO_IDX[m.uci()] for m in board.legal_moves]
                + [-1] * (80 - board.legal_moves.count())
                for board in boards
            ]
        )
        nll = factored_nll(factors, targets, factor_tables())
        metrics = PolicyMetrics()
        metrics.update(joint, targets, legal, torch.tensor([32, 8, 3]), nll=nll)
        result = metrics.result()
        self.assertAlmostEqual(result["loss"], float(nll.mean()), places=6)
        self.assertIn("factor_valid_mass", result)
        self.assertLessEqual(result["factor_valid_mass"], 1.0 + 1e-5)

    def test_wide_readout_roundtrip(self):
        import tempfile
        from pathlib import Path

        config = {
            **CONFIG,
            "readout": "wide",
            "orientation": "side_to_move",
            "layout": "pieces",
        }
        model = create_model(config)
        ids = torch.from_numpy(encode_boards(FENS))
        logits, _ = model(ids=ids)
        self.assertEqual(logits.shape, (len(FENS), len(IDX_TO_UCI_MOVE)))
        self.assertTrue(torch.isfinite(logits).all())
        logits.square().mean().backward()
        self.assertGreater(float(model.act_tokens.grad.abs().sum()), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wide.pth"
            save_checkpoint(path, model.eval(), config)
            torch.testing.assert_close(
                load_model(path)(ids=ids)[0], model.eval()(ids=ids)[0], rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
