"""Prepare roots, curate traces, or fit a vocabulary and encode a dataset."""

import argparse
import json
from pathlib import Path

from pondera.data.config import DEFAULT_CONFIG_PATH, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Provision sources and prepare roots")
    prepare.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="SOURCE=PATH",
        help="Use a local input for a configured source",
    )
    prepare.add_argument("--cache", type=Path)
    curate = commands.add_parser("curate", help="Produce traces from a prepared pool")
    curate.add_argument("--input", required=True, type=Path)
    curate.add_argument("--engine", default="stockfish")
    encode = commands.add_parser(
        "encode", help="Fit a merge vocabulary and encode traces"
    )
    encode.add_argument("--input", required=True, type=Path)
    pack = commands.add_parser(
        "pack", help="Pack a tokenized dataset into training streams"
    )
    pack.add_argument("--input", required=True, type=Path)
    for command in (prepare, curate, encode, pack):
        command.add_argument("--config", default=DEFAULT_CONFIG_PATH, type=Path)
        command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        from pondera.data.roots import prepare_pool

        inputs = {}
        for value in args.input:
            name, separator, path = value.partition("=")
            if not separator or not name or not path or name in inputs:
                parser.error("--input requires distinct SOURCE=PATH values")
            inputs[name] = Path(path)
        manifest = prepare_pool(config, args.output, inputs=inputs, cache=args.cache)
        result = {
            "root_pool": str(args.output / "roots.jsonl"),
            "roots": sum(manifest["counts"].values()),
        }
    elif args.command == "curate":
        from pondera.data.pipeline import run_pipeline

        result = run_pipeline(config, args.input, args.output, engine=args.engine)
        result = {key: value for key, value in result.items() if key != "statistics"}
    elif args.command == "pack":
        from pondera.data.packing import write_packed

        result = write_packed(args.input, args.output, seed=config.seed)
    else:
        from pondera.data.tokenize import merge_and_encode

        result = merge_and_encode(args.input, args.output, config.merge)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
