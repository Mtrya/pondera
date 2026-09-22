"""Evaluate a checkpoint against Stockfish using paired fastchess games."""

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sf-depth", required=True, type=int)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--fastchess", default="fastchess")
    parser.add_argument("--stockfish", default="stockfish")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--openings", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    from pondera.evaluation.match import run_match

    output = args.output_dir or Path(".local/runs") / datetime.now(
        timezone.utc
    ).strftime("match-%Y%m%d-%H%M%S-%f")
    result = run_match(
        args.checkpoint,
        args.sf_depth,
        output,
        args.games,
        args.concurrency,
        args.fastchess,
        args.stockfish,
        args.openings,
        args.device,
    )
    print(json.dumps(asdict(result), indent=2))
