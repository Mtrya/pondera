"""UCI serving for packed-trainer checkpoints with optional reasoning."""

import argparse
from functools import partial

import torch

from pondera.chess import _rules as rules
from pondera.chess.symmetry import reflect_uci
from pondera.chess.tracker import repetition_count
from pondera.chess.vocabulary import MOVE, UCI_MOVE_TO_IDX
from pondera.inference.trace import load_checkpoint
from pondera.inference.uci import serve


def choose_legal_move(logits, fen, legal_moves=None):
    """Select a legal action from one vocabulary-wide root decision."""
    position = rules.Board(fen)
    if legal_moves is None:
        legal_moves = [move.uci() for move in position.legal_moves]
    candidates = [
        (MOVE + UCI_MOVE_TO_IDX[reflect_uci(uci) if not position.turn else uci], uci)
        for uci in legal_moves
        if uci in UCI_MOVE_TO_IDX
    ]
    if not candidates:
        raise ValueError("No legal move has a vocabulary token")
    tokens = torch.tensor([token for token, _ in candidates], device=logits.device)
    return candidates[int(logits[tokens].float().argmax())][1]


def pick_with_reasoning(predictor, board, device, *, reasoning_tokens):
    if board.is_game_over():
        return "0000"
    fen = board.fen()
    logits = predictor.decision_logits(
        fen, repetition_count(board), reasoning_tokens=reasoning_tokens
    )
    return choose_legal_move(logits, fen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reasoning-tokens", type=int, default=0)
    parser.add_argument("--loop-repetitions", type=int)
    args = parser.parse_args()
    if args.reasoning_tokens < 0:
        parser.error("--reasoning-tokens must be nonnegative")
    picker = partial(pick_with_reasoning, reasoning_tokens=args.reasoning_tokens)
    load = partial(load_checkpoint, loop_repetitions=args.loop_repetitions)
    if args.device.split(":", 1)[0] == "npu":
        import torch_npu

        torch_npu.npu.set_device(args.device)
    serve(args.checkpoint, args.device, load=load, pick=picker)


if __name__ == "__main__":
    main()
