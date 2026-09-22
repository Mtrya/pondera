"""Policy accuracy, cross entropy, and probability assigned to illegal moves."""

import chess
import torch

from pondera.chess.mapping import UCI_MOVE_TO_IDX
from pondera.chess.position import legal_move_indices
from pondera.chess.tokenize import encode_fens


@torch.inference_mode()
def evaluate_policy(model, rows, device="cpu", batch_size=128):
    model.eval()
    totals = {"positions": 0, "loss_sum": 0.0, "correct": 0, "invalid_sum": 0.0}

    def consume(batch):
        fens = [row["fen"] for row in batch]
        ids = torch.from_numpy(encode_fens(fens)).to(device)
        targets = torch.tensor(
            [UCI_MOVE_TO_IDX[row["next_move"]] for row in batch], device=device
        )
        logits, _ = model(ids=ids)
        logits = logits.float()
        totals["loss_sum"] += torch.nn.functional.cross_entropy(
            logits, targets, reduction="sum"
        ).item()
        totals["correct"] += (logits.argmax(-1) == targets).sum().item()
        probabilities = logits.softmax(-1)
        for i, fen in enumerate(fens):
            board = chess.Board(fen)
            legal = legal_move_indices(board)
            if board.can_claim_draw():
                legal.append(UCI_MOVE_TO_IDX["<claim_draw>"])
            totals["invalid_sum"] += (1 - probabilities[i, legal].sum()).item()
        totals["positions"] += len(batch)

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            consume(batch)
            batch = []
    if batch:
        consume(batch)
    count = totals["positions"]
    if not count:
        raise ValueError("Evaluation dataset is empty")
    return {
        "positions": count,
        "act_loss": totals["loss_sum"] / count,
        "top1": totals["correct"] / count,
        "invalid_move_prob": totals["invalid_sum"] / count,
    }
