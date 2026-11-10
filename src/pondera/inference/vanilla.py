"""Zero-presentation generation: board ++ FINAL, then the decision token.

Prefill reuses the training encoding path (encode_board ++ a standalone
FINAL, which merge fitting structurally keeps out of every composite), so
the split the model sees at inference is exactly the one it trained on.
The next-token distribution is masked to the position's legal moves and
sampled at temperature zero.
"""

import numpy as np
import torch

from pondera.chess.codec import encode_board
from pondera.chess.symmetry import reflect_uci
from pondera.chess.vocabulary import FINAL, MOVE, UCI_MOVE_TO_IDX
from pondera.inference.trace import load_checkpoint


def load_vanilla_checkpoint(path, device="cpu"):
    """Load either packed-trainer model into eval mode."""
    return load_checkpoint(path, device).model


@torch.no_grad()
def choose_move(model, fen, repetition=1, legal_moves=None):
    """Return the argmax legal UCI decision for one root position."""
    import chess

    board = encode_board(fen, repetition)
    device = model.token_embedding.weight.device
    prefill = torch.as_tensor(
        np.concatenate([board, np.asarray([FINAL], dtype=board.dtype)]).astype(
            np.int64
        ),
        device=device,
    )
    cu_seqlens = torch.tensor([0, prefill.numel()], dtype=torch.int32, device=device)
    logits, _ = model.forward_cached(prefill, cu_seqlens)
    scores = logits[-1].float()
    position = chess.Board(fen)
    if legal_moves is None:
        legal_moves = [move.uci() for move in position.legal_moves]
    candidates = [
        (MOVE + UCI_MOVE_TO_IDX[reflect_uci(uci) if not position.turn else uci], uci)
        for uci in legal_moves
        if uci in UCI_MOVE_TO_IDX
    ]
    if not candidates:
        raise ValueError("No legal move has a vocabulary token")
    tokens = torch.tensor([token for token, _ in candidates], device=device)
    return candidates[int(scores[tokens].argmax())][1]
