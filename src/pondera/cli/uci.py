"""Expose a checkpoint as a UCI engine."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from pondera.inference.uci import serve

    serve(args.checkpoint, args.device)


if __name__ == "__main__":
    main()
