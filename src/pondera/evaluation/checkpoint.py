"""Evaluate unconstrained checkpoint answers on fixed roots and chess puzzles."""

import hashlib
import heapq
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median

import pyarrow.parquet as pq

from pondera.chess import _rules as rules
from pondera.chess.tracker import repetition_count
from pondera.data.dataset import parquet_sources, write_manifest
from pondera.data.roots.pool import MANIFEST_FILE, POOL_FILE, RootPool
from pondera.evaluation.reference import Reference
from pondera.inference.trace import load_checkpoint


def _rows(path, puzzle):
    path = Path(path)
    candidate = path / POOL_FILE if path.is_dir() else path
    if (
        not puzzle
        and candidate.name == POOL_FILE
        and candidate.with_name(MANIFEST_FILE).is_file()
        and candidate.is_file()
    ):
        yield from RootPool(candidate).iter_range()
        return
    if path.is_dir() and (path / "roots.jsonl").exists():
        path /= "roots.jsonl"
    if path.suffix == ".jsonl":
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".json":
        values = json.loads(path.read_text())
        yield from values if isinstance(values, list) else [values]
    else:
        for file, _, _ in parquet_sources(path):
            parquet = pq.ParquetFile(file)
            wanted = (
                {"PuzzleId", "FEN", "Moves", "Rating"}
                if puzzle
                else {"fen", "repetition", "root_id"}
            )
            columns = [name for name in parquet.schema_arrow.names if name in wanted]
            for batch in parquet.iter_batches(
                columns=columns, batch_size=128, use_threads=False
            ):
                yield from batch.to_pylist()


def select_rows(path, *, puzzle, limit, seed):
    """Bounded, order-independent sampling of unique roots or puzzle IDs."""
    selected, heap = {}, []
    for row in _rows(path, puzzle):
        if puzzle:
            board = rules.Board(row["FEN"])
            moves = row["Moves"].split()
            if len(moves) < 2:
                raise ValueError("A puzzle needs a setup move and a solution")
            board.push_uci(moves[0])
            if not board.is_valid() or board.is_game_over():
                raise ValueError("Puzzle setup must produce a valid playable root")
            key = str(row.get("PuzzleId") or (row["FEN"] + "|" + row["Moves"]))
            int(row["Rating"])
        else:
            board = rules.Board(row["fen"])
            repetition = row.get("repetition", 1)
            if type(repetition) is not int or repetition < 1:
                raise ValueError("Root repetition must be a positive integer")
            if not board.is_valid() or board.is_game_over() or repetition >= 5:
                raise ValueError("Evaluation roots must be valid playable positions")
            key = f"{board.fen(en_passant='fen')}|{repetition}"
        if key in selected:
            continue
        priority = int.from_bytes(
            hashlib.sha256(f"{seed}:{key}".encode()).digest(), "big"
        )
        item = (-priority, key)
        if len(heap) == limit:
            if item <= heap[0]:
                continue
            _, discarded = heapq.heappop(heap)
            del selected[discarded]
        heapq.heappush(heap, item)
        selected[key] = row
    return [(key, selected[key]) for _, key in sorted(heap, reverse=True)]


def _generation(predictor, reference, fen, repetition, config, budget):
    generated = predictor.generate(
        fen, repetition, reasoning_tokens=budget, answer_tokens=config.answer_tokens
    )
    result = asdict(generated)
    result["answer"]["valid"] = generated.answer.valid
    result["decision_quality"] = (
        reference.grade_decision(fen, repetition, generated.answer.decision)
        if generated.answer.move_legal
        else None
    )
    result["quality"] = (
        reference.grade(fen, repetition, generated.answer, pv_plies=config.pv_plies)
        if generated.answer.valid
        else None
    )
    return result


def _puzzle(predictor, reference, row, config, budget):
    board = rules.Board(row["FEN"])
    moves = row["Moves"].split()
    board.push_uci(moves[0])
    turns = []
    first = whole = False
    for position in range(1, len(moves), 2):
        generation = _generation(
            predictor, reference, board.fen(), repetition_count(board), config, budget
        )
        turns.append({"fen": board.fen(), **generation})
        answer = generation["answer"]
        if not answer["move_legal"] or answer["decision"] == "<claim_draw>":
            break
        candidate = rules.Move.from_uci(answer["decision"])
        board.push(candidate)
        # Lichess accepts every checkmating alternative, including mate-in-one.
        correct = candidate.uci() == moves[position] or board.is_checkmate()
        if position == 1:
            first = correct
        if not correct:
            break
        if board.is_checkmate() or position + 1 == len(moves):
            whole = True
            break
        if position + 2 >= len(moves):
            whole = True
            break
        board.push_uci(moves[position + 1])
    return {
        "rating": int(row["Rating"]),
        "first_correct": first,
        "whole_correct": whole,
        "turns": turns,
    }


def _summary(records, *, puzzle=False):
    generations = [
        record["generation"] if record["kind"] == "root" else record["turns"][0]
        for record in records
    ]
    answers = [generation["answer"] for generation in generations]
    qualities = [
        generation["quality"]
        for generation in generations
        if generation["quality"] is not None
    ]
    decisions = [
        generation["decision_quality"]
        for generation in generations
        if generation["decision_quality"] is not None
    ]
    count = len(records)
    result = {
        "count": count,
        "decision_quality_count": len(decisions),
        "decision_failure_count": count - len(decisions),
        "answer_quality_count": len(qualities),
        "errors": dict(
            Counter(error for answer in answers for error in answer["errors"])
        ),
    }
    for name in (
        "final_found",
        "end_found",
        "format_valid",
        "move_legal",
        "pv_legal",
        "valid",
    ):
        result[name + "_rate"] = (
            sum(answer[name] for answer in answers) / count if count else None
        )
    for name in (
        "cp_loss",
        "expected_score_loss",
        "move_rank_fraction",
        "best_move",
    ):
        values = [quality[name] for quality in decisions if quality[name] is not None]
        result[name] = {
            "count": len(values),
            "mean": mean(values) if values else None,
            "median": median(values) if values else None,
        }
    for name in (
        "value_absolute_error",
        "value_bin_correct",
        "mate_distance_correct",
    ):
        values = [quality[name] for quality in qualities if quality[name] is not None]
        result[name] = {
            "count": len(values),
            "mean": mean(values) if values else None,
            "median": median(values) if values else None,
        }
    all_generations = [
        generation
        for record in records
        for generation in (
            [record["generation"]] if record["kind"] == "root" else record["turns"]
        )
    ]
    result["generation_count"] = len(all_generations)
    result["generated_tokens"] = sum(
        generation["generated_tokens"] for generation in all_generations
    )
    result["expanded_atoms"] = sum(
        generation["expanded_atoms"] for generation in all_generations
    )
    result["mean_generation_seconds"] = (
        mean(generation["seconds"] for generation in all_generations)
        if all_generations
        else None
    )
    result["mean_pv_plies"] = (
        mean(len(answer["pv"]) for answer in answers) if count else None
    )
    for root_player, name in (
        (True, "root_player_pv_cp_loss"),
        (False, "opponent_pv_cp_loss"),
    ):
        losses = [
            ply["cp_loss"]
            for quality in qualities
            for ply in quality["pv"]
            if ply["root_player"] == root_player and ply["cp_loss"] is not None
        ]
        result[name] = {"count": len(losses), "mean": mean(losses) if losses else None}
    puzzles = [record for record in records if record["kind"] == "puzzle"]
    if puzzles or puzzle:
        result["first_accuracy"] = (
            sum(record["first_correct"] for record in puzzles) / len(puzzles)
            if puzzles
            else None
        )
        result["whole_accuracy"] = (
            sum(record["whole_correct"] for record in puzzles) / len(puzzles)
            if puzzles
            else None
        )
    return result


def evaluate_checkpoint(
    checkpoint,
    output,
    config,
    *,
    roots=None,
    puzzles=None,
    engine="stockfish",
    cache=None,
    device="cpu",
):
    """Write aggregate and per-position results; no training or telemetry calls."""
    if roots is None and puzzles is None:
        raise ValueError("Provide roots, puzzles, or both")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    inputs = {}
    for kind, path in (("root", roots), ("puzzle", puzzles)):
        if path is not None:
            inputs[kind] = select_rows(
                path, puzzle=kind == "puzzle", limit=config.limit, seed=config.seed
            )
            if not inputs[kind]:
                raise ValueError(f"No {kind} records selected")
    predictor = load_checkpoint(checkpoint, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    budgets = sorted({0, config.reasoning_tokens})
    output.mkdir()
    reference_path = Path(cache) if cache is not None else output / "reference.sqlite"
    report = {
        "status": "running",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_metadata": predictor.metadata,
        "config": asdict(config),
        "sources": {
            kind: str(Path(path).resolve())
            for kind, path in (("root", roots), ("puzzle", puzzles))
            if path is not None
        },
        "budgets": {},
    }
    reference = None
    try:
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference = Reference(
            engine,
            reference_path,
            nodes=config.nodes,
            threads=config.threads,
            hash_mb=config.hash_mb,
        )
        report["reference"] = reference.settings
        write_manifest(output / "summary.json", report)
        with (output / "positions.jsonl").open("w", buffering=1) as handle:
            for budget in budgets:
                records = []
                for kind, rows in inputs.items():
                    for key, row in rows:
                        record = {
                            "kind": kind,
                            "id": key,
                            "reasoning_tokens": budget,
                        }
                        if kind == "root":
                            fen, repetition = row["fen"], row.get("repetition", 1)
                            # Prepare all candidate scores even if this checkpoint's answer fails.
                            reference.scores(rules.Board(fen))
                            record.update(
                                fen=fen,
                                repetition=repetition,
                                generation=_generation(
                                    predictor,
                                    reference,
                                    fen,
                                    repetition,
                                    config,
                                    budget,
                                ),
                            )
                        else:
                            root = rules.Board(row["FEN"])
                            root.push_uci(row["Moves"].split()[0])
                            reference.scores(rules.Board(root.fen()))
                            record.update(
                                _puzzle(predictor, reference, row, config, budget)
                            )
                        records.append(record)
                        handle.write(json.dumps(record, allow_nan=False) + "\n")
                summary = {"overall": _summary(records)}
                for kind in inputs:
                    summary[kind] = _summary(
                        [record for record in records if record["kind"] == kind],
                        puzzle=kind == "puzzle",
                    )
                if "puzzle" in inputs:
                    bounds = config.rating_bounds
                    summary["puzzle_buckets"] = {}
                    for index, low in enumerate(bounds):
                        high = bounds[index + 1] if index + 1 < len(bounds) else None
                        subset = [
                            record
                            for record in records
                            if record["kind"] == "puzzle"
                            and low <= record["rating"]
                            and (high is None or record["rating"] < high)
                        ]
                        summary["puzzle_buckets"][
                            f"{low}-{high if high is not None else '+'}"
                        ] = _summary(subset, puzzle=True)
                report["budgets"][str(budget)] = summary
                write_manifest(output / "summary.json", report)
        report["reference_cache"] = {
            "hits": reference.hits,
            "searches": reference.searches,
        }
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        write_manifest(output / "summary.json", report)
        raise
    finally:
        if reference is not None:
            reference.close()
    report["status"] = "complete"
    write_manifest(output / "summary.json", report)
    return report
