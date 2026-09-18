"""Run a fastchess match: a Pondera checkpoint vs Stockfish at fixed depth.

Appends one row to results/leaderboard.md. Baseline conditions (see
notes/provisioning.md): SF 17.1 single thread at fixed depth, our side
deterministic (temperature=0, no pondering), fixed sequential opening suite.
"""

import argparse
import datetime
import os
import re
import subprocess

FASTCHESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "fastchess")
REPO = os.path.dirname(os.path.abspath(__file__))
STOCKFISH = "/usr/games/stockfish"
LEADERBOARD = os.path.join(REPO, "results", "leaderboard.md")

HEADER = (
    "| date | checkpoint | opponent | games | W-D-L | score | elo_diff | 95% CI | notes |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
)


def append_leaderboard(row: str) -> None:
    os.makedirs(os.path.dirname(LEADERBOARD), exist_ok=True)
    if not os.path.exists(LEADERBOARD):
        with open(LEADERBOARD, "w") as f:
            f.write("# Leaderboard\n\n")
            f.write(HEADER)
    with open(LEADERBOARD, "a") as f:
        f.write(row + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sf-depth", type=int, required=True)
    parser.add_argument("--games", type=int, default=100, help="total games (even)")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="parallel games; engine processes load the model at startup, "
        "so high concurrency can trip fastchess's uciok timeout",
    )
    parser.add_argument(
        "--openings", default=os.path.join(REPO, "match", "openings.epd")
    )
    parser.add_argument(
        "--device", default=None, help="cuda/cpu; omit to let uci.py auto-detect"
    )
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    rounds = args.games // 2
    ckpt_name = os.path.basename(args.checkpoint.rstrip("/"))
    os.makedirs(os.path.join(REPO, "log"), exist_ok=True)
    pgn = os.path.join(REPO, "log", f"match_{ckpt_name}_d{args.sf_depth}.pgn")

    device_arg = f" --device {args.device}" if args.device else ""
    cmd = [
        FASTCHESS,
        "-engine",
        f"cmd={os.path.join(REPO, '.venv', 'bin', 'python')}",
        f"args={os.path.join(REPO, 'uci.py')} --checkpoint {args.checkpoint}{device_arg}",
        "name=pondera",
        "-engine",
        f"cmd={STOCKFISH}",
        f"name=sf-d{args.sf_depth}",
        f"plies={args.sf_depth}",
        "-each",
        "tc=300+0",
        "timemargin=1000",
        "-rounds",
        str(rounds),
        "-games",
        "2",
        "-concurrency",
        str(args.concurrency),
        "-openings",
        f"file={args.openings}",
        "format=epd",
        "order=sequential",
        "-draw",
        "movenumber=80",
        "movecount=10",
        "score=8",
        "-maxmoves",
        "200",
        "-pgnout",
        f"file={pgn}",
        "notation=uci",
        "-srand",
        "42",
    ]

    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    tail = "\n".join(out.strip().splitlines()[-15:])
    print(tail)

    score_m = re.findall(
        r"Games: (\d+), Wins: (\d+), Losses: (\d+), Draws: (\d+)", out
    )
    elo_m = re.findall(r"\nElo: (-?[\d.]+|nan) \+/- ([\d.]+|nan)", out)
    if not score_m or not elo_m:
        print("ERROR: could not parse fastchess result, not writing leaderboard")
        raise SystemExit(1)

    total, w, l, d = (int(x) for x in score_m[-1])
    score = (w + 0.5 * d) / total if total else 0.0
    elo, ci = elo_m[-1]
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    row = (
        f"| {date} | {args.checkpoint} | sf-d{args.sf_depth} | {total} "
        f"| {w}-{d}-{l} | {score:.3f} | {elo} | ±{ci} | {args.notes} |"
    )
    append_leaderboard(row)
    print("Leaderboard updated:", row)


if __name__ == "__main__":
    main()
