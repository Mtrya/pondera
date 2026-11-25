"""Independent python-chess transitions for the frozen piece-state circuit."""

import random
import unittest

import chess
import torch

from pondera.chess.vocabulary import UCI_MOVE_TO_IDX
from pondera.models.board_state import CompiledPieceTransition


def indicators(board):
    result = torch.zeros(64, 12)
    for square, piece in board.piece_map().items():
        result[square, "PNBRQKpnbrqk".index(piece.symbol())] = 1
    return result


class CompiledPieceTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.circuit = CompiledPieceTransition()

    def test_all_legal_moves_match_independent_board_updates(self):
        positions = [
            chess.Board(fen)
            for fen in (
                chess.STARTING_FEN,
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
                "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
                "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
                "1r2k3/P6p/8/8/8/8/p6P/4K3 w - - 0 1",
                "4k3/P6p/8/8/8/8/p6P/1R2K3 b - - 0 1",
            )
        ]
        rng = random.Random(640)
        board = chess.Board()
        for ply in range(256):
            if board.is_game_over():
                board = chess.Board()
            board.push(rng.choice(list(board.legal_moves)))
            if ply % 8 == 0:
                positions.append(board.copy())
        before, moves, after = [], [], []
        for position in positions:
            for move in position.legal_moves:
                before.append(indicators(position))
                moves.append(UCI_MOVE_TO_IDX[move.uci()])
                advanced = position.copy()
                advanced.push(move)
                after.append(indicators(advanced))
        states, moves, expected = (
            torch.stack(before),
            torch.tensor(moves),
            torch.stack(after),
        )
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                output = self.circuit.to(dtype)(states.to(dtype), moves)
                torch.testing.assert_close(output.float(), expected, atol=0, rtol=0)

    def test_claim_preserves_placement_and_one_position_supports_arbitrary_batch_axes(
        self,
    ):
        circuit = self.circuit.float()
        state = indicators(chess.Board())
        claim = torch.tensor(UCI_MOVE_TO_IDX["<claim_draw>"])
        torch.testing.assert_close(circuit(state, claim), state, atol=0, rtol=0)
        batched = state.expand(2, 3, 64, 12)
        torch.testing.assert_close(
            circuit(batched, claim.expand(2, 3)), batched, atol=0, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
