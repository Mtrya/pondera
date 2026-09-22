"""Train a policy model using a reproducible TOML configuration."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path(".local/data/policy"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Stop at this absolute optimizer step, including steps before resume",
    )
    parser.add_argument("--track", action="store_true", help="Log metrics to SwanLab")
    args = parser.parse_args()
    from pondera.training.supervised import load_config, train

    print(
        train(
            load_config(args.config),
            args.data_dir,
            args.output_dir,
            args.device,
            args.resume,
            args.max_steps,
            args.track,
        )
    )
