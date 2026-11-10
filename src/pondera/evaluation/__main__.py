"""Evaluate checkpoint playing strength on roots and puzzles."""

import argparse
import json
from pathlib import Path

from pondera.evaluation.checkpoint import evaluate_checkpoint
from pondera.evaluation.config import DEFAULT_CONFIG_PATH, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--roots", type=Path)
    parser.add_argument("--puzzles", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--engine", default="stockfish")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = evaluate_checkpoint(
        args.checkpoint,
        args.output,
        load_config(args.config),
        roots=args.roots,
        puzzles=args.puzzles,
        engine=args.engine,
        cache=args.cache,
        device=args.device,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
