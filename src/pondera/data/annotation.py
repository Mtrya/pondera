"""Offline Stockfish annotations with scores at each completed search depth."""

import json
import math
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import file_digest, sha256
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
    max_time: float | None = None


def analyse(engine, board, config: AnnotationConfig) -> dict:
    by_depth = {}
    completed = {}
    expected = min(config.multipv, board.legal_moves.count())
    nodes = None
    reported_depth = 0
    with engine.analysis(
        board,
        chess.engine.Limit(depth=config.depth, time=config.max_time),
        multipv=config.multipv,
        game=object(),
        options={"UCI_ShowWDL": True},
    ) as stream:
        for info in stream:
            nodes = info.get("nodes", nodes)
            if "depth" not in info or not info.get("pv") or "score" not in info:
                continue
            depth, rank = info["depth"], info.get("multipv", 1)
            reported_depth = max(reported_depth, depth)
            ranks = by_depth.setdefault(depth, {})
            if rank == 1:
                ranks.clear()
            if info.get("lowerbound") or info.get("upperbound"):
                ranks.pop(rank, None)
                continue
            score = info["score"].pov(board.turn)
            wdl = info["wdl"].pov(board.turn)
            ranks[rank] = {
                "bm": info["pv"][0].uci(),
                "cp": score.score(),
                "mate": score.mate(),
                "pv": [move.uci() for move in info["pv"]],
                "wdl": [wdl.wins, wdl.draws, wdl.losses],
                "value": (wdl.wins - wdl.losses) / wdl.total(),
            }
            if (
                set(ranks) == set(range(1, expected + 1))
                and len({entry["bm"] for entry in ranks.values()}) == expected
            ):
                completed[depth] = {
                    "d": depth,
                    **{f"r{k}": value for k, value in sorted(ranks.items())},
                }
    return {
        "per_depth": [row for _, row in sorted(completed.items())],
        "reached_depth": max(completed, default=0),
        "reported_depth": reported_depth,
        "requested_depth_completed": config.depth in completed,
        "incomplete_depths": sorted(set(by_depth) - set(completed)),
        "expected_candidates": expected,
        "nodes": nodes,
    }


def _position_id(record):
    position = [record["fen"], record.get("repetition", 1)]
    return sha256(json.dumps(position).encode()).hexdigest()


def _annotate(record, engine, config):
    board = chess.Board(record["fen"])
    if not board.is_valid():
        raise ValueError(f"Invalid position: {record['fen']}")
    result = {
        **record,
        "position_id": _position_id(record),
        "root_player": "white" if board.turn else "black",
        "repetition": record.get("repetition", 1),
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
            done.add(_position_id(json.loads(line)))
    return done


def annotate_positions(records, output, config=AnnotationConfig(), workers=1):
    if min(
        config.depth, config.multipv, config.threads, config.hash_mb, workers
    ) < 1 or (
        config.max_time is not None
        and (not math.isfinite(config.max_time) or config.max_time <= 0)
    ):
        raise ValueError("Annotation limits and worker count must be positive")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".manifest.json")
    with chess.engine.SimpleEngine.popen_uci(config.engine) as engine:
        identity = dict(engine.id)
        defaults = {name: option.default for name, option in engine.options.items()}
    engine_path = Path(shutil.which(config.engine) or config.engine).resolve()
    with engine_path.open("rb") as handle:
        checksum = file_digest(handle, "sha256").hexdigest()
    manifest = {
        "format_version": 2,
        "teacher": identity,
        "engine_sha256": checksum,
        "engine_defaults": defaults,
        "config": asdict(config),
        "score_perspective": "root_player",
        "value_scale": "(wdl_wins - wdl_losses) / wdl_total",
        "history": "FEN only; no preceding game moves supplied to the teacher",
        "search_state": "new game for every position",
    }
    if output.exists():
        if load_manifest(manifest_path) != manifest:
            raise ValueError("Annotation configuration differs from existing output")
        done = _completed_fens(output)
    else:
        done = set()
        write_manifest(manifest_path, manifest)

    def pending():
        for record in records:
            key = _position_id(record)
            if key not in done:
                done.add(key)
                yield record

    engines = []
    worker = local()

    def annotate(record):
        if not hasattr(worker, "engine"):
            worker.engine = chess.engine.SimpleEngine.popen_uci(config.engine)
            engines.append(worker.engine)
            worker.engine.configure({"Threads": config.threads, "Hash": config.hash_mb})
        return _annotate(record, worker.engine, config)

    count = 0
    try:
        # Each worker owns a Stockfish process; Python threads only coordinate I/O.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            with output.open("a") as handle:
                # Bound queued work and memory while preserving deterministic order.
                iterator = iter(pending())
                futures = deque()
                for _ in range(2 * workers):
                    record = next(iterator, None)
                    if record is not None:
                        futures.append(pool.submit(annotate, record))
                while futures:
                    record = futures.popleft().result()
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    count += 1
                    following = next(iterator, None)
                    if following is not None:
                        futures.append(pool.submit(annotate, following))
    finally:
        for engine in engines:
            engine.quit()
    return count
