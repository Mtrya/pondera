"""Deterministic, search-free policy inference."""

import chess
import torch

from pondera.chess.codec import encode_board, legal_move_indices
from pondera.chess.tracker import repetition_count
from pondera.chess.vocabulary import IDX_TO_UCI_MOVE
from pondera.models.policy import DepthPolicyModel


@torch.inference_mode()
def pick_move(model, board: chess.Board, device="cpu") -> str:
    if not isinstance(model, DepthPolicyModel):
        raise ValueError("UCI move selection supports depth-policy models only")
    if board.is_game_over():
        return "0000"
    ids = torch.from_numpy(
        encode_board(board.fen(en_passant="fen"), repetition_count(board))
    )
    logits, _ = model(ids=ids.unsqueeze(0).to(device))
    legal = legal_move_indices(board)
    index = logits[0, legal].float().argmax().item()
    return IDX_TO_UCI_MOVE[legal[index]]
