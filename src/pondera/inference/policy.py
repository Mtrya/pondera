"""Deterministic, search-free policy inference."""

import chess
import torch

from pondera.chess.codec import encode_board, legal_move_indices
from pondera.chess.domain import repetition_count
from pondera.chess.vocabulary import IDX_TO_UCI_MOVE


@torch.inference_mode()
def pick_move(model, board: chess.Board, device="cpu") -> str:
    if board.is_game_over():
        return "0000"
    ids = torch.from_numpy(encode_board(board.fen(), repetition_count(board)))
    logits, _ = model(ids=ids.unsqueeze(0).to(device))
    legal = legal_move_indices(board)
    index = logits[0, legal].float().argmax().item()
    return IDX_TO_UCI_MOVE[legal[index]]
