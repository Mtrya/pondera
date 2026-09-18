"""UCI adapter: expose a PonderaModel checkpoint as a UCI engine.

Usage:
    .venv/bin/python uci.py --checkpoint kaupane/ChessFormer-SL [--device cuda]

Baseline inference is deterministic: argmax over legal moves, no pondering,
temperature 0 (see AGENTS.md). "<claim_draw>" is never emitted; draws are left
to the match runner's adjudication. `go` parameters are ignored — inference is
fixed-compute by design.
"""

import argparse
import sys

import chess
import numpy as np
import torch

from chess_core import IDX_TO_UCI_MOVE, UCI_MOVE_TO_IDX, encode_fen
from evaluate import load_model


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def repetition_count(board: chess.Board) -> int:
    if board.is_repetition(3):
        return 3
    if board.is_repetition(2):
        return 2
    return 1


@torch.no_grad()
def pick_move(model, board: chess.Board, device: torch.device) -> str:
    ids = torch.from_numpy(
        encode_fen(board.fen(), repetition_count(board)).astype(np.int64)
    ).unsqueeze(0)
    logits, _ = model(ids=ids.to(device))
    logits = logits[0].float()

    mask = torch.full_like(logits, float("-inf"))
    legal_ids = [UCI_MOVE_TO_IDX[m.uci()] for m in board.legal_moves]
    mask[legal_ids] = 0.0

    move_idx = (logits + mask).argmax().item()
    return IDX_TO_UCI_MOVE[move_idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    # The model is loaded lazily on the first `go` so that the UCI handshake
    # (uciok / readyok) always responds instantly; loading here would race the
    # match runner's engine-startup timeout when checkpoints are large or come
    # from the network.
    model = None
    board = chess.Board()

    def send(msg: str) -> None:
        print(msg, flush=True)

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        parts = line.strip().split()
        if not parts:
            continue
        cmd = parts[0]

        if cmd == "uci":
            send(f"id name Pondera ({args.checkpoint})")
            send("id author Pondera Project")
            send("uciok")
        elif cmd == "isready":
            send("readyok")
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd == "position":
            try:
                if parts[1] == "startpos":
                    board = chess.Board()
                    idx = 2
                elif parts[1] == "fen":
                    idx = 2
                    fen_fields = []
                    while idx < len(parts) and parts[idx] != "moves":
                        fen_fields.append(parts[idx])
                        idx += 1
                    board = chess.Board(" ".join(fen_fields))
                else:
                    continue
                if idx < len(parts) and parts[idx] == "moves":
                    for uci in parts[idx + 1 :]:
                        board.push_uci(uci)
            except Exception as e:
                log(f"position parse error: {e}")
        elif cmd == "go":
            if model is None:
                model = load_model(args.checkpoint, device)
            if board.is_game_over():
                send("bestmove 0000")
            else:
                send(f"bestmove {pick_move(model, board, device)}")
        elif cmd == "stop":
            pass
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
