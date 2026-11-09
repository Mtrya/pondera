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
from pondera.chess.vocabulary import FINAL, MOVE, UCI_MOVE_TO_IDX
from pondera.models.linear import LinearVanillaModel
from pondera.models.vanilla import VanillaModel

MODEL_CLASSES = {"vanilla": VanillaModel, "linear": LinearVanillaModel}


def load_vanilla_checkpoint(path, device="cpu"):
    """Load either packed-trainer model into eval mode."""
    state = torch.load(path, map_location=device, weights_only=True)
    model = MODEL_CLASSES[state["arch"]](**state["model_config"])
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval()


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
    if legal_moves is None:
        legal_moves = [move.uci() for move in chess.Board(fen).legal_moves]
    candidates = [
        (MOVE + UCI_MOVE_TO_IDX[uci], uci)
        for uci in legal_moves
        if uci in UCI_MOVE_TO_IDX
    ]
    if not candidates:
        raise ValueError("No legal move has a vocabulary token")
    token, uci = max(candidates, key=lambda pair: float(scores[pair[0]]))
    return uci
