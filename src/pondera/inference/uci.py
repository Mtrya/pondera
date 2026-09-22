"""UCI protocol for fixed-compute policy inference."""

import sys

import chess

from pondera.inference.policy import pick_move
from pondera.models.checkpoint import load_model


def parse_position(parts: list[str]) -> chess.Board:
    if parts[0] == "startpos":
        board, index = chess.Board(), 1
    elif parts[0] == "fen":
        board, index = chess.Board(" ".join(parts[1:7])), 7
    else:
        raise ValueError(f"Unknown position type: {parts[0]}")
    if index < len(parts):
        if parts[index] != "moves":
            raise ValueError("Expected moves after position")
        for move in parts[index + 1 :]:
            board.push_uci(move)
    return board


def serve(checkpoint: str, device: str) -> None:
    model = None
    board = chess.Board()
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        command = parts[0]
        if command == "uci":
            print("id name Pondera\nid author Pondera Project\nuciok", flush=True)
        elif command == "isready":
            print("readyok", flush=True)
        elif command == "ucinewgame":
            board = chess.Board()
        elif command == "position":
            board = parse_position(parts[1:])
        elif command == "go":
            # Keep startup independent of checkpoint I/O and reserve stdout for UCI.
            if model is None:
                from contextlib import redirect_stdout

                with redirect_stdout(sys.stderr):
                    model = load_model(checkpoint, device)
            print(f"bestmove {pick_move(model, board, device)}", flush=True)
        elif command == "quit":
            break
