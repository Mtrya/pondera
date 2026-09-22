"""Deterministic, search-free policy inference."""

import chess
import torch

from pondera.chess.mapping import IDX_TO_UCI_MOVE
from pondera.chess.position import legal_move_indices, repetition_count
from pondera.chess.tokenize import encode_fen


@torch.inference_mode()
def pick_move(model, board: chess.Board, device="cpu") -> str:
    if board.is_game_over():
        return "0000"
    ids = torch.from_numpy(encode_fen(board.fen(), repetition_count(board)))
    logits, _ = model(ids=ids.unsqueeze(0).to(device))
    legal = legal_move_indices(board)
    index = logits[0, legal].float().argmax().item()
    return IDX_TO_UCI_MOVE[legal[index]]
