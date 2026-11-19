"""UCI serving for packed-trainer checkpoints with optional reasoning."""

import argparse
from functools import partial

from pondera.chess.tracker import repetition_count
from pondera.inference.trace import load_checkpoint
from pondera.inference.uci import serve
from pondera.inference.vanilla import (
    choose_legal_move,
    choose_move,
    load_vanilla_checkpoint,
)


def pick(model, board, device):
    if board.is_game_over():
        return "0000"
    return choose_move(model, board.fen(), repetition=repetition_count(board))


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
    if args.reasoning_tokens:
        picker = partial(pick_with_reasoning, reasoning_tokens=args.reasoning_tokens)
        load = partial(load_checkpoint, loop_repetitions=args.loop_repetitions)
    else:
        picker = pick
        load = partial(load_vanilla_checkpoint, loop_repetitions=args.loop_repetitions)
    serve(args.checkpoint, args.device, load=load, pick=picker)


if __name__ == "__main__":
    main()
