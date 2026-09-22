"""Run paired fastchess matches and retain complete machine-readable results."""

import datetime
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True)
class MatchResult:
    games: int
    wins: int
    draws: int
    losses: int
    score: float
    elo_diff: float | None
    elo_ci: float | None


def parse_result(output: str, expected_games: int) -> MatchResult:
    scores = re.findall(
        r"Games: (\d+), Wins: (\d+), Losses: (\d+), Draws: (\d+)", output
    )
    elos = re.findall(r"\nElo: (-?[\d.]+|nan) \+/- ([\d.]+|nan)", output)
    if not scores or not elos:
        raise ValueError("Missing fastchess result")
    games, wins, losses, draws = map(int, scores[-1])
    if games != expected_games or wins + losses + draws != games:
        raise ValueError(f"Incomplete match: {games}/{expected_games} games")
    elo, ci = (float(value) for value in elos[-1])
    return MatchResult(
        games,
        wins,
        draws,
        losses,
        (wins + 0.5 * draws) / games,
        elo if math.isfinite(elo) else None,
        ci if math.isfinite(ci) else None,
    )


def run_match(
    checkpoint,
    sf_depth,
    output_dir,
    games=100,
    concurrency=4,
    fastchess="fastchess",
    stockfish="stockfish",
    openings=None,
    device="cpu",
):
    if games < 2 or games % 2 or sf_depth < 1 or concurrency < 1:
        raise ValueError(
            "Use an even positive game count, positive depth and concurrency"
        )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    openings = (
        Path(openings).resolve()
        if openings
        else Path(str(files("pondera.evaluation").joinpath("openings.epd")))
    )
    if Path(checkpoint).exists():
        checkpoint = str(Path(checkpoint).resolve())
    if Path(fastchess).exists():
        fastchess = str(Path(fastchess).resolve())
    if Path(stockfish).exists():
        stockfish = str(Path(stockfish).resolve())
    command = [
        str(fastchess),
        "-engine",
        f"cmd={sys.executable}",
        "args="
        + shlex.join(
            [
                "-m",
                "pondera.cli.uci",
                "--checkpoint",
                str(checkpoint),
                "--device",
                device,
            ]
        ),
        "name=pondera",
        "-engine",
        f"cmd={stockfish}",
        f"name=sf-d{sf_depth}",
        f"plies={sf_depth}",
        "option.Threads=1",
        "option.Hash=16",
        "-each",
        "tc=300+0",
        "timemargin=1000",
        "-rounds",
        str(games // 2),
        "-games",
        "2",
        "-concurrency",
        str(concurrency),
        "-openings",
        f"file={openings}",
        "format=epd",
        "order=sequential",
        "-draw",
        "movenumber=80",
        "movecount=10",
        "score=8",
        "-maxmoves",
        "200",
        "-pgnout",
        f"file={output_dir / 'games.pgn'}",
        "notation=uci",
        "-srand",
        "42",
    ]
    metadata = {
        "checkpoint": str(checkpoint),
        "opponent_depth": sf_depth,
        "games_requested": games,
        "device": device,
        "search": False,
        "pondering": False,
        "temperature": 0,
        "openings_sha256": hashlib.sha256(openings.read_bytes()).hexdigest(),
        "command": command,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    process = subprocess.run(command, capture_output=True, text=True, cwd=output_dir)
    output = process.stdout + process.stderr
    (output_dir / "fastchess.log").write_text(output)
    process.check_returncode()
    result = parse_result(output, games)
    (output_dir / "result.json").write_text(
        json.dumps(asdict(result), indent=2, allow_nan=False) + "\n"
    )
    return result
