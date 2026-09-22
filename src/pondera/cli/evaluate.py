"""Evaluate policy metrics on local parquet position files."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from pondera.data.parquet import sampled_batches
    from pondera.evaluation.policy import evaluate_policy
    from pondera.models.checkpoint import load_model

    rows = (
        row
        for batch in sampled_batches(args.files, ["fen", "next_move"], args.rows)
        for row in batch.to_pylist()
    )
    print(
        json.dumps(
            evaluate_policy(
                load_model(args.checkpoint, args.device),
                rows,
                args.device,
                args.batch_size,
            ),
            indent=2,
        )
    )
