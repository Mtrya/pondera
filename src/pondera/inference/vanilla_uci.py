"""UCI serving for vanilla packed-trainer checkpoints (zero presentation)."""

import argparse

from pondera.chess.tracker import repetition_count
from pondera.inference.uci import serve
from pondera.inference.vanilla import choose_move, load_vanilla_checkpoint


def pick(model, board, device):
    if board.is_game_over():
        return "0000"
    return choose_move(model, board.fen(), repetition=repetition_count(board))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    serve(args.checkpoint, args.device, load=load_vanilla_checkpoint, pick=pick)


if __name__ == "__main__":
    main()
