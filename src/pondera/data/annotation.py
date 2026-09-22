"""Offline Stockfish annotations with scores at each completed search depth."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import local

import chess
import chess.engine

from pondera.data.storage import load_manifest, write_manifest


@dataclass(frozen=True)
class AnnotationConfig:
    engine: str = "stockfish"
    depth: int = 16
    multipv: int = 5
    threads: int = 1
    hash_mb: int = 64
    max_time: float = 15.0


def analyse(engine, board, config: AnnotationConfig) -> dict:
    by_depth = {}
    nodes = None
    with engine.analysis(
        board,
        chess.engine.Limit(depth=config.depth, time=config.max_time),
        multipv=config.multipv,
    ) as stream:
        for info in stream:
            if "depth" not in info or not info.get("pv") or "score" not in info:
                continue
            score = info["score"].white()
            by_depth.setdefault(info["depth"], {})[info.get("multipv", 1)] = {
                "bm": info["pv"][0].uci(),
                "cp": score.score(),
                "mate": score.mate(),
                "pv": [move.uci() for move in info["pv"]],
            }
            nodes = info.get("nodes", nodes)
    reached = max(by_depth, default=0)
    per_depth = []
    for depth, ranks in sorted(by_depth.items()):
        row = {"d": depth}
        for rank, entry in sorted(ranks.items()):
            row[f"r{rank}"] = {
                key: value
                for key, value in entry.items()
                if key != "pv" or depth == reached
            }
        per_depth.append(row)
    return {"per_depth": per_depth, "reached_depth": reached, "nodes": nodes}


def _annotate(record, engine, config):
    board = chess.Board(record["fen"])
    result = {
        **record,
        "pieces": chess.popcount(board.occupied),
        "fullmove": board.fullmove_number,
    }
    if board.is_game_over():
        return {**result, "terminal": True}
    start = time.monotonic()
    result["unit"] = analyse(engine, board, config)
    result["elapsed_ms"] = int((time.monotonic() - start) * 1000)
    return result


def _completed_fens(output):
    """Read committed JSONL records, discarding an unterminated final write."""
    done = set()
    with output.open("r+b") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                handle.truncate(start)
                break
            done.add(json.loads(line)["fen"])
    return done


def annotate_positions(records, output, config=AnnotationConfig(), workers=1):
    if (
        min(config.depth, config.multipv, config.threads, config.hash_mb, workers) < 1
        or config.max_time <= 0
    ):
        raise ValueError("Annotation limits and worker count must be positive")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".manifest.json")
    with chess.engine.SimpleEngine.popen_uci(config.engine) as engine:
        identity = dict(engine.id)
    manifest = {"format_version": 1, "teacher": identity, "config": asdict(config)}
    if output.exists():
        if load_manifest(manifest_path) != manifest:
            raise ValueError("Annotation configuration differs from existing output")
        done = _completed_fens(output)
    else:
        done = set()
        write_manifest(manifest_path, manifest)
    pending = []
    for record in records:
        if record["fen"] not in done:
            pending.append(record)
            done.add(record["fen"])
    engines = []
    worker = local()

    def annotate(record):
        if not hasattr(worker, "engine"):
            worker.engine = chess.engine.SimpleEngine.popen_uci(config.engine)
            engines.append(worker.engine)
            worker.engine.configure({"Threads": config.threads, "Hash": config.hash_mb})
        return _annotate(record, worker.engine, config)

    try:
        # Each worker owns a Stockfish process; Python threads only coordinate I/O.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            with output.open("a") as handle:
                for record in pool.map(annotate, pending):
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
    finally:
        for engine in engines:
            engine.quit()
    return len(pending)
