"""Offline trace production: configuration, engine execution, shards, and checks.

The public entry point is ``run_pipeline``. It prepares roots when necessary,
then runs engine -> sidecar validation -> synthesis -> replay in memory for each
root and publishes complete shards atomically. No sidecar directory or separate
collection stage is required.
"""

import argparse
import fcntl
import json
import math
import os
import random
import re
import select
import shutil
import statistics
import subprocess
import time
import tomllib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

from pondera.chess.codec import encode_board, value_bin
from pondera.chess.symmetry import reflect_uci
from pondera.chess.vocabulary import (
    CHESS_TOKENIZER_VERSION,
    DEPTH,
    IDX_TO_UCI_MOVE,
    MATE_LOSS,
    MATE_WIN,
    MOVE,
    PAD,
    VALUE,
)
from pondera.data.dataset import (
    DATA_ARRAYS,
    DATA_FORMAT_VERSION,
    STATE_ARRAYS,
    load_manifest,
    load_trace_arrays,
    validate_trace_arrays,
    write_manifest,
)
from pondera.data.trace import (
    CLAIM_DRAW_UCI,
    DRAW_CLAIM_BOUND,
    PRESENTATIONS,
    ROLE_COUNT,
    SIDECAR_SCHEMA,
    SIDECAR_SCHEMA_WIDE,
    TRACE_GRAMMAR_VERSION,
    SynthesisError,
    mate_distance_moves,
    parse_sidecar,
    replay,
    synthesize_root,
    validate_root,
)

DEFAULT_CONFIG_PATH = Path("configs/trace.toml")
PRESENTATION_NAMES = ("zero", "full", "random")
_DIGITS = re.compile(r"^[1-9][0-9]*$")


def _integer(section, name, value, minimum=None):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"[{section}] {name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"[{section}] {name} must be at least {minimum}")
    return value


def _probability(section, name, value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"[{section}] {name} must be a number")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"[{section}] {name} must lie in [0, 1]")
    return value


def _distribution(section, name, value, key_parser):
    if isinstance(value, (int, str)) and not isinstance(value, bool):
        value = {value: 1.0}
    if not isinstance(value, dict) or not value:
        raise ValueError(f"[{section}] {name} must be a value or a nonempty table")
    result = {}
    for raw_key, probability in value.items():
        key = key_parser(raw_key)
        if key in result:
            raise ValueError(f"[{section}] {name} repeats a key")
        result[key] = _probability(section, f"{name}.{raw_key}", probability)
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-9):
        raise ValueError(f"[{section}] {name} probabilities must sum to 1")
    return tuple(sorted(result.items(), key=lambda item: str(item[0])))


def _positive_key(value):
    if isinstance(value, bool):
        raise ValueError("distribution keys must be positive integers")
    if isinstance(value, int):
        if value < 1:
            raise ValueError("distribution keys must be positive integers")
        return value
    if not isinstance(value, str) or not _DIGITS.match(value):
        raise ValueError("distribution keys must be positive integers")
    return int(value)


def _presentation_key(value):
    if value not in PRESENTATION_NAMES:
        raise ValueError(f"presentation must be one of {PRESENTATION_NAMES}")
    return value


@dataclass(frozen=True)
class PathsConfig:
    archive: str
    roots: str
    output: str
    engine: str


@dataclass(frozen=True)
class RootsConfig:
    train: int
    val: int
    claimable_share: float


@dataclass(frozen=True)
class SearchConfig:
    depth: int
    teacher_width: int
    hash_mb: int


@dataclass(frozen=True)
class SynthesisConfig:
    claim_draw: bool
    board_descent_hazard: float
    board_frontier_p: float
    presentations: tuple
    random_min_fraction: float
    random_max_fraction: float

    @property
    def zero(self):
        return dict(self.presentations).get("zero", 0.0)

    @property
    def full(self):
        return dict(self.presentations).get("full", 0.0)

    @property
    def random(self):
        return dict(self.presentations).get("random", 0.0)


@dataclass(frozen=True)
class ReleaseConfig:
    shard_size: int
    workers: int


@dataclass(frozen=True)
class SamplingConfig:
    widths: tuple = None
    depths: tuple = None
    presentations: tuple = None

    def identity(self):
        payload = {}
        for name, distribution in (
            ("widths", self.widths),
            ("depths", self.depths),
            ("presentations", self.presentations),
        ):
            if distribution is not None:
                payload[name] = {
                    str(key): probability for key, probability in distribution
                }
        return payload


@dataclass(frozen=True)
class TraceConfig:
    seed: int
    paths: PathsConfig
    roots: RootsConfig
    search: SearchConfig
    synthesis: SynthesisConfig
    release: ReleaseConfig
    sampling: SamplingConfig

    def search_dict(self):
        return {
            "depth": self.search.depth,
            "teacher_width": self.search.teacher_width,
            "hash_mb": self.search.hash_mb,
        }

    def board_dict(self):
        return {
            "board_descent_hazard": self.synthesis.board_descent_hazard,
            "board_frontier_p": self.synthesis.board_frontier_p,
        }

    def presentation_dict(self):
        return {
            "zero": self.synthesis.zero,
            "full": self.synthesis.full,
            "random": self.synthesis.random,
            "last_root": 0.0,
            "random_min_fraction": self.synthesis.random_min_fraction,
            "random_max_fraction": self.synthesis.random_max_fraction,
        }

    def synthesis_dict(self):
        return {
            "confirmatory_repetition": "faithful",
            "claim_draw": self.synthesis.claim_draw,
        }


def sample_distribution(rng, distribution):
    roll = rng.random()
    cumulative = 0.0
    fallback = distribution[0][0]
    for key, probability in distribution:
        if probability > 0.0:
            fallback = key
        cumulative += probability
        if roll < cumulative:
            return key
    return fallback


def sample_width(sampling, seed, root_id, default_width):
    if sampling is None or sampling.widths is None:
        return default_width
    return sample_distribution(
        random.Random(f"{seed}:{root_id}:width"), sampling.widths
    )


def sample_depth(sampling, seed, root_id, default_depth):
    if sampling is None or sampling.depths is None:
        return default_depth
    return sample_distribution(
        random.Random(f"{seed}:{root_id}:depth"), sampling.depths
    )


def sample_presentation(sampling, seed, root_id):
    if sampling is None or sampling.presentations is None:
        return None
    return sample_distribution(
        random.Random(f"{seed}:{root_id}:presentation"), sampling.presentations
    )


def parse_recipe(values):
    if not isinstance(values, dict):
        raise ValueError("recipe must be a table")
    unknown = set(values) - {"widths", "depths", "presentations"}
    if unknown or not values:
        raise ValueError("recipe has an unknown or empty distribution")
    return SamplingConfig(
        widths=_distribution("recipe", "widths", values["widths"], _positive_key)
        if "widths" in values
        else None,
        depths=_distribution("recipe", "depths", values["depths"], _positive_key)
        if "depths" in values
        else None,
        presentations=_distribution(
            "recipe", "presentations", values["presentations"], _presentation_key
        )
        if "presentations" in values
        else None,
    )


def _table(values, name, required):
    if not isinstance(values, dict):
        raise ValueError(f"[{name}] must be a table")
    unknown = set(values) - set(required)
    if unknown:
        raise ValueError(f"Unknown config key in [{name}]: {sorted(unknown)[0]}")
    missing = set(required) - set(values)
    if missing:
        raise ValueError(f"Missing config key in [{name}]: {sorted(missing)[0]}")
    return values


def load_config(path=DEFAULT_CONFIG_PATH):
    path = Path(path)
    with path.open("rb") as handle:
        values = tomllib.load(handle)
    allowed = {"seed", "paths", "roots", "search", "synthesis", "release"}
    if set(values) - allowed:
        raise ValueError(f"Unknown config section: {sorted(set(values) - allowed)[0]}")
    for section in allowed - {"seed"}:
        if section not in values:
            raise ValueError(f"Missing config section: [{section}]")
    seed = _integer("seed", "seed", values.get("seed", 640))

    paths = _table(values["paths"], "paths", {"archive", "roots", "output", "engine"})
    for name in ("archive", "roots", "output", "engine"):
        if not isinstance(paths[name], str) or (
            name in {"roots", "output", "engine"} and not paths[name]
        ):
            raise ValueError(f"[paths] {name} must be a string")
    paths = PathsConfig(**paths)

    roots = _table(values["roots"], "roots", {"train", "val", "claimable_share"})
    train = _integer("roots", "train", roots["train"], 0)
    val = _integer("roots", "val", roots["val"], 0)
    if train + val < 1:
        raise ValueError("[roots] train + val must be positive")
    roots = RootsConfig(
        train, val, _probability("roots", "claimable_share", roots["claimable_share"])
    )

    search = _table(values["search"], "search", {"hash_mb", "widths", "depths"})
    hash_mb = _integer("search", "hash_mb", search["hash_mb"], 1)
    widths = _distribution("search", "widths", search["widths"], _positive_key)
    depths = _distribution("search", "depths", search["depths"], _positive_key)

    synthesis = _table(
        values["synthesis"],
        "synthesis",
        {
            "claim_draw",
            "board_descent_hazard",
            "board_frontier_p",
            "presentations",
            "random_min_fraction",
            "random_max_fraction",
        },
    )
    if not isinstance(synthesis["claim_draw"], bool):
        raise ValueError("[synthesis] claim_draw must be a boolean")
    presentations = _distribution(
        "synthesis", "presentations", synthesis["presentations"], _presentation_key
    )
    minimum = _probability(
        "synthesis", "random_min_fraction", synthesis["random_min_fraction"]
    )
    maximum = _probability(
        "synthesis", "random_max_fraction", synthesis["random_max_fraction"]
    )
    if minimum >= maximum:
        raise ValueError("[synthesis] random cut interval must be nonempty")
    synthesis = SynthesisConfig(
        synthesis["claim_draw"],
        _probability(
            "synthesis", "board_descent_hazard", synthesis["board_descent_hazard"]
        ),
        _probability("synthesis", "board_frontier_p", synthesis["board_frontier_p"]),
        presentations,
        minimum,
        maximum,
    )

    release = _table(values["release"], "release", {"shard_size", "workers"})
    shard_size = _integer("release", "shard_size", release["shard_size"], 1)
    workers = _integer("release", "workers", release["workers"], 1)
    release = ReleaseConfig(shard_size, workers)
    search = SearchConfig(depths[0][0], widths[0][0], hash_mb)
    sampling = SamplingConfig(widths, depths, presentations)
    return TraceConfig(seed, paths, roots, search, synthesis, release, sampling)


def _percentiles(values):
    if not values:
        return {"count": 0, "min": 0, "p10": 0.0, "median": 0.0, "p90": 0.0, "max": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": int(min(values)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": int(max(values)),
        "mean": float(statistics.mean(values)),
    }


class RootProfile:
    """Streaming aggregate of synthesized-root statistics.

    Roots are consumed one at a time, so a release can compute its statistics
    without retaining the token arrays of every completed shard.
    """

    def __init__(self, presentations):
        self.presentations = presentations
        self.roots = 0
        self.rows = 0
        self.transitions = 0
        self.overturns = 0
        self.refutations = 0
        self.rounds = 0
        self.full_lengths = []
        self.total_lengths = []
        self.presentation_lengths = {name: [] for name in PRESENTATIONS}
        self.presentation_thinking = {name: [] for name in PRESENTATIONS}
        self.presentation_counts = Counter()
        self.rounds_distribution = Counter()
        self.converged_distribution = Counter()
        self.refutation_distribution = Counter()
        self.value_shapes = Counter()
        self.splits = Counter()
        self.roots_with_overturn = 0
        self.first_round_matches_final = 0
        self.all_draw_roots = 0
        self.claim_draw_roots = 0
        self.claimable_roots = 0
        self.board_descent = 0
        self.board_frontier = 0
        self.exact_evals = 0
        self.terminal_evals = 0
        self.frontier_evals = 0
        self.omitted_bound = 0
        self.omitted_early_refutation = 0
        self.omitted_root_fail_high = 0
        self.per_root = []

    def add(self, root):
        """Fold one synthesized root into the aggregate."""
        root_id = root.root["root_id"]
        rows = [
            (name, len(root.rows[name].tokens), root.rows[name].thinking_tokens)
            for name in self.presentations[root_id]
        ]
        self._add(root_id, root.root["split"], root.stats, rows)

    def add_stored(self, row, stats, token_count):
        self._add(
            row["root_id"],
            row["split"],
            stats,
            [(row["presentation"], token_count, row["thinking_tokens"])],
        )

    def _add(self, root_id, split, stats, rows):
        self.roots += 1
        self.rows += len(rows)
        self.transitions += max(stats["rounds"] - 1, 0)
        self.overturns += stats["overturned_rounds"]
        self.refutations += stats["refutations"]
        self.rounds += stats["rounds"]
        self.full_lengths.append(stats["thinking_tokens"])
        self.total_lengths.append(stats["total_tokens"])
        for name, token_count, thinking_tokens in rows:
            self.presentation_lengths[name].append(token_count)
            self.presentation_thinking[name].append(thinking_tokens)
            self.presentation_counts[name] += 1
        self.rounds_distribution[stats["rounds"]] += 1
        self.converged_distribution[stats["converged_depth"]] += 1
        for count in stats["refutations_per_round"]:
            self.refutation_distribution[count] += 1
        values = [value for value in stats["root_values"] if value is not None]
        if not values:
            self.value_shapes["mate_only"] += 1
        else:
            changes = sum(1 for left, right in zip(values, values[1:]) if left != right)
            if changes == 0:
                self.value_shapes["flat"] += 1
            elif all(right >= left for left, right in zip(values, values[1:])):
                self.value_shapes["monotone_up"] += 1
            elif all(right <= left for left, right in zip(values, values[1:])):
                self.value_shapes["monotone_down"] += 1
            else:
                self.value_shapes["mixed"] += 1
        self.splits[split] += 1
        self.roots_with_overturn += stats["overturned_rounds"] > 0
        self.first_round_matches_final += stats["first_round_matches_final"]
        self.all_draw_roots += stats["all_draw"]
        self.claim_draw_roots += stats["claim_draw"]
        self.claimable_roots += stats["claimable"]
        self.board_descent += stats["board_descent"]
        self.board_frontier += stats["board_frontier"]
        self.exact_evals += stats["exact_evals"]
        self.terminal_evals += stats["terminal_evals"]
        self.frontier_evals += stats["frontier_evals"]
        self.omitted_bound += stats["omitted_bound"]
        self.omitted_early_refutation += stats["omitted_early_refutation"]
        self.omitted_root_fail_high += stats["omitted_root_fail_high"]
        self.per_root.append(
            {
                "root_id": root_id,
                "split": split,
                "rounds": stats["rounds"],
                "thinking_tokens": stats["thinking_tokens"],
                "overturned_rounds": stats["overturned_rounds"],
                "all_draw": stats["all_draw"],
                "claim_draw": stats["claim_draw"],
            }
        )

    def result(self):
        """The statistics document consumed by release manifests."""
        if not self.roots:
            raise ValueError("A statistics profile needs at least one root")
        return {
            "roots": self.roots,
            "rows": self.rows,
            "split_counts": dict(sorted(self.splits.items())),
            "presentation_counts": {
                name: self.presentation_counts[name] for name in PRESENTATIONS
            },
            "rounds": dict(sorted(self.rounds_distribution.items())),
            "rounds_summary": _percentiles([root["rounds"] for root in self.per_root]),
            "thinking_tokens": _percentiles(self.full_lengths),
            "total_tokens": _percentiles(self.total_lengths),
            "presentation_lengths": {
                name: _percentiles(values)
                for name, values in self.presentation_lengths.items()
            },
            "presentation_thinking_tokens": {
                name: _percentiles(values)
                for name, values in self.presentation_thinking.items()
            },
            "overturned_rounds": self.overturns,
            "overturn_rate": self.overturns / self.transitions
            if self.transitions
            else 0.0,
            "roots_with_overturn": self.roots_with_overturn,
            "refutations": self.refutations,
            "refutations_per_round": self.refutations / max(self.rounds, 1),
            "refutations_per_round_distribution": dict(
                sorted(self.refutation_distribution.items())
            ),
            "converged_depth": dict(sorted(self.converged_distribution.items())),
            "first_round_matches_final": self.first_round_matches_final,
            "value_trajectory_shapes": dict(sorted(self.value_shapes.items())),
            "all_draw_roots": self.all_draw_roots,
            "claim_draw_roots": self.claim_draw_roots,
            "claimable_roots": self.claimable_roots,
            "board_descent": self.board_descent,
            "board_frontier": self.board_frontier,
            "exact_evals": self.exact_evals,
            "terminal_evals": self.terminal_evals,
            "frontier_evals": self.frontier_evals,
            "omitted_bound": self.omitted_bound,
            "omitted_early_refutation": self.omitted_early_refutation,
            "omitted_root_fail_high": self.omitted_root_fail_high,
            "per_root": self.per_root,
        }


def profile_roots(roots, presentations=None):
    """Aggregate the statistics over synthesized roots and their written rows."""
    roots = list(roots)
    if not roots:
        raise ValueError("A statistics profile needs at least one root")
    if presentations is None:
        presentations = {root.root["root_id"]: PRESENTATIONS for root in roots}
    profile = RootProfile(presentations)
    for root in roots:
        profile.add(root)
    return profile.result()


class CollectionError(RuntimeError):
    """A pipeline input or result that violates the production contract."""


def _root_teacher(root, config):
    width = root.get("teacher_width", config.search.teacher_width)
    depth = root.get("depth", config.search.depth)
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise CollectionError(f"Root {root['root_id']} has an invalid teacher width")
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        raise CollectionError(f"Root {root['root_id']} has an invalid depth")
    return width, depth


def _engine_path(config):
    path = Path(config.paths.engine)
    if path.is_file():
        return path.resolve()
    resolved = shutil.which(config.paths.engine)
    if resolved is None:
        raise FileNotFoundError(config.paths.engine)
    return Path(resolved).resolve()


def _check_sidecar(root, sidecar, config, engine_name, teacher=None):
    width, depth = _root_teacher(root, config)
    repetition = validate_root(root, sidecar)
    if sidecar.engine != engine_name:
        raise CollectionError(f"Engine name differs for root {root['root_id']}")
    if teacher is not None and (
        sidecar.value_mate != teacher[0] or sidecar.max_ply != teacher[1]
    ):
        raise CollectionError(f"Teacher constants differ for root {root['root_id']}")
    if sidecar.search_start["depth"] != depth:
        raise CollectionError(f"Requested depth differs for root {root['root_id']}")
    if sidecar.search_end["depth"] != depth:
        raise CollectionError(f"Search did not complete for root {root['root_id']}")
    if sidecar.session["options"]["threads"] != 1:
        raise CollectionError(f"Thread count differs for root {root['root_id']}")
    if sidecar.session["options"]["hash_mb"] != config.search.hash_mb:
        raise CollectionError(f"Hash size differs for root {root['root_id']}")
    if sidecar.configured_width != width:
        raise CollectionError(f"Teacher width differs for root {root['root_id']}")
    return repetition


def _run_engine(engine_path, fen, config, trace_path, width, depth, timeout=900):
    """Drive one engine over raw UCI in the contract's exact command order.

    python-chess manages MultiPV itself and refuses to configure it, so the
    collector speaks plain UCI: options first, TraceLog last, a readiness
    barrier, then position and a depth-limited go.
    """
    process = subprocess.Popen(
        [str(engine_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        bufsize=0,
    )

    def send(line):
        process.stdin.write((line + "\n").encode())
        process.stdin.flush()

    buffer = b""

    def read_line(deadline):
        nonlocal buffer
        while b"\n" not in buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CollectionError("engine timed out")
            ready, _, _ = select.select([process.stdout], [], [], remaining)
            if not ready:
                continue
            chunk = process.stdout.read(65536)
            if not chunk:
                raise CollectionError("engine closed its output")
            buffer += chunk
        line, buffer = buffer.split(b"\n", 1)
        return line.decode(errors="replace").strip()

    def read_until(prefix, deadline):
        seen = []
        while True:
            line = read_line(deadline)
            seen.append(line)
            if line.startswith(prefix):
                return line, seen

    deadline = time.monotonic() + timeout
    try:
        send("uci")
        _, banner = read_until("uciok", deadline)
        if not any(line.startswith("option name TraceLog ") for line in banner):
            raise CollectionError(f"Engine does not expose TraceLog: {engine_path}")
        send(f"setoption name Threads value {1}")
        send(f"setoption name Hash value {config.search.hash_mb}")
        send(f"setoption name MultiPV value {width}")
        send(f"setoption name TraceLog value {trace_path}")
        send("isready")
        read_until("readyok", deadline)
        send(f"position fen {fen}")
        send(f"go depth {depth}")
        bestmove, _ = read_until("bestmove", deadline)
    finally:
        try:
            send("quit")
        except (BrokenPipeError, ValueError):
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    parts = bestmove.split()
    if len(parts) < 2:
        raise CollectionError(f"engine returned no bestmove: {bestmove!r}")
    return parts[1]


def _selection_record(root):
    return {
        key: value
        for key, value in root.items()
        if key not in {"teacher_width", "depth", "presentation"}
    }


RELEASE_ONLY_METADATA = frozenset({"root_selection"})
ROOT_ROW_FIELDS = ("split", "fen", "repetition", "turn", "position_key", "source")
CHAMPION_FIELDS = ("round", "depth", "move", "raw", "wdl", "mate", "value_token", "pv")


def _write_jsonl(path, rows):
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _array(directory, name, rows, width, dtype, padding):
    array = np.lib.format.open_memmap(
        directory / f"{name}.npy", mode="w+", dtype=dtype, shape=(len(rows), width)
    )
    array[:] = padding
    for index, values in enumerate(rows):
        array[index, : len(values)] = values
    array.flush()
    return array


def _one_dimensional(directory, name, values, dtype):
    array = np.lib.format.open_memmap(
        directory / f"{name}.npy", mode="w+", dtype=dtype, shape=(len(values),)
    )
    array[:] = values
    array.flush()
    return array


def _root_row(root, synthesized, presentation):
    row = synthesized.rows[presentation]
    payload = {
        "root_id": root["root_id"],
        "presentation": presentation,
        "teacher_width": synthesized.stats["teacher_width"],
        "depth": synthesized.stats["depth"],
        "rounds": row.rounds,
        "thinking_tokens": row.thinking_tokens,
        "final_move": row.final_move,
        "final_value_bin": row.final_value_bin,
        "mate_distance": row.mate_distance,
        "claim_draw": row.claim_draw,
    }
    for name in ROOT_ROW_FIELDS:
        if name in root:
            payload[name] = root[name]
    return payload


def _champion_row(root, synthesized):
    rounds = []
    for audit in synthesized.round_audits:
        entry = {
            "round": audit.round,
            "depth": audit.depth,
            "move": audit.move,
            "raw": audit.raw,
            "wdl": None if audit.wdl is None else list(audit.wdl),
            "mate": audit.mate,
            "value_token": audit.value_token,
            "pv": list(audit.pv),
            "refutations": audit.refutations,
            "candidates": [
                {
                    "move": candidate.move,
                    "raw": candidate.raw,
                    "wdl": None if candidate.wdl is None else list(candidate.wdl),
                    "mate": candidate.mate,
                    "value_token": candidate.value_token,
                    "order": candidate.order,
                    "draw": candidate.draw,
                }
                for candidate in audit.candidates
            ],
        }
        rounds.append(entry)
    return {
        "root_id": root["root_id"],
        "rounds": rounds,
        "stats": synthesized.stats,
    }


def _write_shard(directory, records, metadata, presentations):
    boards, traces, roles, lengths = [], [], [], []
    no_tool, tool, best, basis, values, counts = [], [], [], [], [], []
    root_rows = []
    champion_rows = []
    for synthesized in records:
        root = synthesized.root
        board = encode_board(root["fen"], int(root["repetition"]))
        for presentation in presentations[root["root_id"]]:
            row = synthesized.rows[presentation]
            boards.append(board)
            traces.append(row.tokens)
            roles.append(row.roles)
            lengths.append(len(row.tokens))
            no_tool.append([boundary.no_tool_end for boundary in row.boundaries])
            tool.append([boundary.tool_end for boundary in row.boundaries])
            best.append([boundary.best for boundary in row.boundaries])
            basis.append([boundary.basis for boundary in row.boundaries])
            values.append([boundary.value for boundary in row.boundaries])
            counts.append(len(row.boundaries))
            root_rows.append(_root_row(root, synthesized, presentation))
        champion_rows.append(_champion_row(root, synthesized))
    if not boards:
        raise ValueError("A trace shard cannot be empty")
    directory.mkdir(parents=True)
    _array(
        directory,
        "board_tokens",
        boards,
        max(map(len, boards)),
        np.int16,
        PAD,
    )
    _array(
        directory,
        "trace_tokens",
        traces,
        max(map(len, traces)),
        np.int16,
        PAD,
    )
    _one_dimensional(directory, "trace_len", lengths, np.int32)
    _array(
        directory,
        "trace_roles",
        roles,
        max(map(len, roles)),
        np.int8,
        0,
    )
    for name, rows in [
        ("state_without_board", no_tool),
        ("state_with_board", tool),
        ("state_best", best),
        ("state_basis", basis),
        ("state_value", values),
    ]:
        _array(directory, name, rows, max(map(len, rows)), np.int32, -1)
    _one_dimensional(directory, "state_count", counts, np.int32)
    _write_jsonl(directory / "roots.jsonl", root_rows)
    _write_jsonl(directory / "champions.jsonl", champion_rows)
    payload = {**metadata, "rows": len(boards)}
    write_manifest(directory / "manifest.json", payload)
    return payload


def _expected_root_rows(roots, presentations):
    return {
        (root["root_id"], presentation)
        for root in roots
        for presentation in presentations[root["root_id"]]
    }


def _recover_shard(output, index, roots, metadata, presentations):
    final = output / f"shard-{index:06d}"
    manifest = load_manifest(final / "manifest.json")
    if not isinstance(manifest.get("rows"), int):
        raise FileExistsError(f"Unregistered shard has no valid row count: {final}")
    published = {"path": final.name, "rows": manifest["rows"]}
    _, rows, _ = verify_shard_files(output, published, metadata)
    if {(row["root_id"], row["presentation"]) for row in rows} != _expected_root_rows(
        roots, presentations
    ):
        raise FileExistsError(f"Unregistered shard has different roots: {final}")
    expected = {root["root_id"]: root for root in roots}
    for row in rows:
        root = expected[row["root_id"]]
        for name in (*ROOT_ROW_FIELDS, "teacher_width", "depth"):
            if name in root and row.get(name) != root[name]:
                raise FileExistsError(
                    f"Unregistered shard root {name} differs: {final}"
                )
    return published


def _profile_shard(output, shard, metadata, profile):
    _, rows, arrays = verify_shard_files(output, shard, metadata)
    champions = {
        row["root_id"]: row for row in _read_champions(Path(output) / shard["path"])
    }
    for index, row in enumerate(rows):
        profile.add_stored(
            row,
            champions[row["root_id"]]["stats"],
            int(arrays["trace_len"][index]),
        )


def _publish_shard(output, index, records, metadata, presentations):
    final = output / f"shard-{index:06d}"
    temporary = output / f".shard-{index:06d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    payload = _write_shard(temporary, records, metadata, presentations)
    if final.exists():
        shutil.rmtree(temporary)
        roots = [record.root for record in records]
        return _recover_shard(output, index, roots, metadata, presentations)
    os.replace(temporary, final)
    return {"path": final.name, "rows": payload["rows"]}


def _release_metadata(identity):
    return {
        "format_version": DATA_FORMAT_VERSION,
        "grammar_version": TRACE_GRAMMAR_VERSION,
        "role_count": ROLE_COUNT,
        "tokenizer_version": CHESS_TOKENIZER_VERSION,
        **identity,
    }


FINAL_SUFFIX_OPS = frozenset({"final", "final_evaluate", "final_pv", "end"})
SEARCH_FIELDS = frozenset({"depth", "teacher_width", "hash_mb"})
BOARD_FIELDS = frozenset({"board_descent_hazard", "board_frontier_p"})
PRESENTATION_FIELDS = frozenset(
    {
        "zero",
        "full",
        "random",
        "last_root",
        "random_min_fraction",
        "random_max_fraction",
    }
)
SYNTHESIS_FIELDS = frozenset({"confirmatory_repetition", "claim_draw"})
TEACHER_FIELDS = frozenset({"schema_version", "value_mate", "max_ply"})
ENGINE_FIELDS = frozenset({"name"})
ROOT_SELECTION_FIELDS = frozenset(
    {"source", "algorithm", "seed", "requested", "counts"}
)


def shard_configuration(metadata):
    return {
        key: value
        for key, value in metadata.items()
        if key not in RELEASE_ONLY_METADATA
    }


def _is_int(value, minimum=None):
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    return minimum is None or value >= minimum


def _is_probability(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
    )


def _release_identity_contract(release, root_count):
    metadata = release.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Release has no synthesis identity")
    required = {
        "teacher",
        "engine",
        "search",
        "board",
        "presentation",
        "synthesis",
        "root_selection",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"Release identity is missing {sorted(missing)[0]}")
    if "recipe" in metadata:
        parse_recipe(metadata["recipe"])

    teacher = metadata["teacher"]
    if not isinstance(teacher, dict) or set(teacher) != TEACHER_FIELDS:
        raise ValueError("Release teacher identity is incomplete")
    if teacher["schema_version"] not in (SIDECAR_SCHEMA, SIDECAR_SCHEMA_WIDE):
        raise ValueError("Release teacher schema is unsupported")
    if not _is_int(teacher["value_mate"], 1) or not _is_int(teacher["max_ply"], 1):
        raise ValueError("Release teacher mate constants are invalid")

    engine = metadata["engine"]
    if (
        not isinstance(engine, dict)
        or set(engine) != ENGINE_FIELDS
        or not isinstance(engine["name"], str)
        or not engine["name"]
    ):
        raise ValueError("Release engine identity is incomplete")

    search = metadata["search"]
    if not isinstance(search, dict) or set(search) != SEARCH_FIELDS:
        raise ValueError("Release identity has incomplete search settings")
    if not _is_int(search["depth"], 1) or not _is_int(search["hash_mb"], 1):
        raise ValueError("Release search limits are invalid")
    if not _is_int(search["teacher_width"], 1):
        raise ValueError("Release teacher width is invalid")
    # A recipe samples the width per root, so the fallback search width does
    # not name the sidecar schema; the rows' recorded widths do, and
    # validate_release checks the schema against them after reading a shard.
    if "recipe" not in metadata and (teacher["schema_version"] == SIDECAR_SCHEMA) != (
        search["teacher_width"] == 1
    ):
        raise ValueError("Release teacher schema differs from the teacher width")

    board = metadata["board"]
    if not isinstance(board, dict) or set(board) != BOARD_FIELDS:
        raise ValueError("Release BOARD identity is incomplete")
    if not all(_is_probability(board[name]) for name in BOARD_FIELDS):
        raise ValueError("Release BOARD probabilities are invalid")

    presentation = metadata["presentation"]
    if not isinstance(presentation, dict) or set(presentation) != PRESENTATION_FIELDS:
        raise ValueError("Release presentation identity is incomplete")
    fractions = [presentation[name] for name in ("zero", "full", "random", "last_root")]
    if not all(_is_probability(value) for value in fractions):
        raise ValueError("Release presentation fractions are invalid")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Release presentation fractions do not sum to one")
    if not (
        _is_probability(presentation["random_min_fraction"])
        and _is_probability(presentation["random_max_fraction"])
        and presentation["random_min_fraction"] < presentation["random_max_fraction"]
    ):
        raise ValueError("Release random cut interval is invalid")

    synthesis = metadata["synthesis"]
    if not isinstance(synthesis, dict) or set(synthesis) != SYNTHESIS_FIELDS:
        raise ValueError("Release synthesis identity is incomplete")
    if synthesis["confirmatory_repetition"] != "faithful":
        raise ValueError("Release confirmatory repetitions are not faithful")
    if not isinstance(synthesis["claim_draw"], bool):
        raise ValueError("Release draw claim setting is invalid")

    root_selection = metadata["root_selection"]
    if (
        not isinstance(root_selection, dict)
        or set(root_selection)
        not in (ROOT_SELECTION_FIELDS, ROOT_SELECTION_FIELDS | {"source_counts"})
        or not isinstance(root_selection.get("source"), str)
        or not isinstance(root_selection.get("algorithm"), str)
        or not _is_int(root_selection.get("seed"))
        or not _is_int(root_selection.get("requested"), 1)
        or not isinstance(root_selection.get("counts"), dict)
        or set(root_selection["counts"]) != {"train", "val"}
        or not all(_is_int(value, 0) for value in root_selection["counts"].values())
        or sum(root_selection["counts"].values()) != root_count
    ):
        raise ValueError("Release root selection identity is incomplete")
    source_counts = root_selection.get("source_counts")
    if source_counts is not None and (
        not isinstance(source_counts, dict)
        or set(source_counts) != {"train", "val"}
        or not all(_is_int(value, 0) for value in source_counts.values())
    ):
        raise ValueError("Release source root counts are invalid")

    for key, value in metadata.items():
        if release.get(key) != value:
            raise ValueError(f"Release metadata {key} differs from its identity")
    return metadata


def validate_release_identity(release):
    complete = release.get("complete_root_ids")
    if (
        not isinstance(complete, list)
        or not complete
        or len(set(complete)) != len(complete)
    ):
        raise ValueError("Release root ids must be nonempty and unique")
    return _release_identity_contract(release, len(complete))


def verify_shard_files(release_dir, shard, expected_metadata=None):
    """Verify a published shard and return its manifest, roots, and arrays."""
    directory = Path(release_dir) / shard["path"]
    manifest = load_manifest(directory / "manifest.json")
    if expected_metadata is not None:
        for key, value in shard_configuration(expected_metadata).items():
            if manifest.get(key) != value:
                raise ValueError(f"Shard metadata {key} differs from its release")
    if manifest["rows"] != shard["rows"]:
        raise ValueError(f"Shard row count differs from its release: {directory}")
    roots = [
        json.loads(line)
        for line in (directory / "roots.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(roots) != manifest["rows"]:
        raise ValueError(f"Shard roots do not match its manifest: {directory}")
    arrays = load_trace_arrays(directory, DATA_ARRAYS)
    validate_trace_arrays(arrays, rows=len(roots))
    return manifest, roots, arrays


def _read_champions(directory):
    directory = Path(directory)
    return [
        json.loads(line)
        for line in (directory / "champions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def _move_token(uci, mirrored):
    return MOVE + IDX_TO_UCI_MOVE_REVERSE[reflect_uci(uci) if mirrored else uci]


IDX_TO_UCI_MOVE_REVERSE = {move: index for index, move in IDX_TO_UCI_MOVE.items()}


def _check_champion_round(root_id, entry, board, value_mate, max_ply):
    if set(entry) != {
        "round",
        "depth",
        "move",
        "raw",
        "wdl",
        "mate",
        "value_token",
        "pv",
        "refutations",
        "candidates",
    }:
        raise ValueError(f"Champion record is incomplete: {root_id}")
    round_number = entry["round"]
    if not _is_int(round_number, 1) or not _is_int(entry["depth"], 1):
        raise ValueError(f"Champion round is invalid: {root_id}")
    move = entry["move"]
    if not isinstance(move, str) or chess.Move.from_uci(move) not in board.legal_moves:
        raise ValueError(f"Champion move is illegal: {root_id}")
    pv = entry["pv"]
    if not isinstance(pv, list) or not pv or pv[0] != move:
        raise ValueError(f"Champion principal variation is invalid: {root_id}")
    tracker = chess.Board(board.fen(en_passant="fen"))
    for index, step in enumerate(pv):
        if not isinstance(step, str):
            raise ValueError(f"Champion principal variation has a bad move: {root_id}")
        try:
            tracker.push_uci(step)
        except ValueError as error:
            raise ValueError(
                f"Champion principal variation is illegal: {root_id}"
            ) from error
        if index + 1 < len(pv) and tracker.is_game_over():
            # The search rule, not python-chess's material rule, defines the
            # trace's terminals; the writer re-checks the exact rule on replay.
            continue
    value_token = entry["value_token"]
    if not isinstance(value_token, int) or not VALUE <= value_token < PAD:
        raise ValueError(f"Champion value token is invalid: {root_id}")
    wdl = entry["wdl"]
    mate = entry["mate"]
    if wdl is None:
        if mate is None or not isinstance(mate, int) or mate == 0:
            raise ValueError(f"Champion mate claim is incomplete: {root_id}")
        expected = mate_token_for(mate)
        if value_token != expected:
            raise ValueError(f"Champion mate token disagrees: {root_id}")
        if mate_distance_moves(entry["raw"], value_mate, max_ply) != mate:
            raise ValueError(f"Champion mate distance disagrees: {root_id}")
    else:
        if (
            not isinstance(wdl, list)
            or len(wdl) != 3
            or any(not _is_int(value, 0) for value in wdl)
            or sum(wdl) != 1000
            or mate is not None
        ):
            raise ValueError(f"Champion WDL is invalid: {root_id}")
        expected = VALUE + value_bin((wdl[0] - wdl[2]) / 1000)
        if value_token != expected:
            raise ValueError(f"Champion value token disagrees: {root_id}")
    candidates = entry["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Champion round has no exact candidates: {root_id}")
    seen_moves = set()
    champion_raw = None
    best_raw = None
    for candidate in candidates:
        if set(candidate) != {
            "move",
            "raw",
            "wdl",
            "mate",
            "value_token",
            "order",
            "draw",
        }:
            raise ValueError(f"Champion candidate is incomplete: {root_id}")
        if (
            not isinstance(candidate["move"], str)
            or chess.Move.from_uci(candidate["move"]) not in board.legal_moves
            or not _is_int(candidate["order"], 1)
            or candidate["move"] in seen_moves
        ):
            raise ValueError(f"Champion candidate is invalid: {root_id}")
        seen_moves.add(candidate["move"])
        if not isinstance(candidate["draw"], bool):
            raise ValueError(f"Champion candidate draw flag is invalid: {root_id}")
        if candidate["draw"]:
            if candidate["wdl"] is not None or candidate["mate"] is not None:
                raise ValueError(
                    f"Champion draw candidate carries an engine value: {root_id}"
                )
            expected = VALUE + value_bin(0.0)
        elif candidate["wdl"] is None:
            if candidate["mate"] is None:
                raise ValueError(f"Champion candidate has no value: {root_id}")
            expected = mate_token_for(candidate["mate"])
        else:
            if (
                not isinstance(candidate["wdl"], list)
                or len(candidate["wdl"]) != 3
                or any(not _is_int(value, 0) for value in candidate["wdl"])
                or sum(candidate["wdl"]) != 1000
                or candidate["mate"] is not None
            ):
                raise ValueError(f"Champion candidate WDL is invalid: {root_id}")
            expected = VALUE + value_bin(
                (candidate["wdl"][0] - candidate["wdl"][2]) / 1000
            )
        if candidate["value_token"] != expected:
            raise ValueError(f"Champion candidate token disagrees: {root_id}")
        if candidate["move"] == move:
            champion_raw = candidate["raw"]
        if best_raw is None or candidate["raw"] > best_raw:
            best_raw = candidate["raw"]
    # Equal root values are possible; the engine's rank names one of the tied
    # maximums, and its tie-break is not carried by the trace.
    if champion_raw is None:
        raise ValueError(f"Champion move is not a written candidate: {root_id}")
    if champion_raw != entry["raw"]:
        raise ValueError(f"Champion raw disagrees with its candidate: {root_id}")
    if best_raw != champion_raw:
        raise ValueError(
            f"Champion is not a raw maximum among the candidates: {root_id}"
        )


def mate_token_for(mate):
    if mate >= 0:
        return MATE_WIN + mate
    return MATE_LOSS + (-mate)


def _expected_states(program, tokens, champions, mirrored):
    """Derive the champion visible at each thought boundary from the trace."""
    entries = {entry["round"]: entry for entry in champions["rounds"]}
    states = []
    round_number = 0
    round_candidates = {}
    round_trace = {}
    round_count = 0
    previous = None
    current = None
    for step in program.steps:
        if step.operation in FINAL_SUFFIX_OPS:
            break
        if step.operation == "depth":
            round_number += 1
            round_candidates = {}
            round_trace = {}
            round_count = 0
            current = previous
        elif step.operation == "evaluate" and len(step.path) == 1:
            move = step.path[-1]
            entry = entries.get(round_number)
            if entry is not None:
                candidate = next(
                    (
                        candidate
                        for candidate in entry["candidates"]
                        if candidate["move"] == move
                    ),
                    None,
                )
                if candidate is not None:
                    trace = round_trace.get(move)
                    if trace is None:
                        round_count += 1
                        trace = round_count
                        round_trace[move] = trace
                    round_candidates[move] = (trace, candidate)
                    best = max(
                        round_candidates.values(),
                        key=lambda item: (item[1]["raw"], -item[0]),
                    )[1]
                    current = (
                        _move_token(best["move"], mirrored),
                        best["value_token"],
                        best["raw"],
                    )
        elif step.operation == "evaluate" and not step.path:
            entry = entries.get(round_number)
            if entry is not None:
                best = (
                    _move_token(entry["move"], mirrored)
                    if current is None or entry["raw"] > current[2]
                    else current[0]
                )
                current = (best, entry["value_token"], entry["raw"])
            previous = current
        if current is None:
            states.append((-1, -1, round_number))
        else:
            states.append((current[0], current[1], round_number))
    return states


def _validate_shard_rows(
    release_dir,
    shard,
    presentations,
    default_width,
    default_depth,
    recipe,
    seed,
    value_mate,
    max_ply,
    expected_metadata=None,
    expected_roots=None,
):
    directory = Path(release_dir) / shard["path"]
    manifest, rows, arrays = verify_shard_files(release_dir, shard, expected_metadata)
    champions_rows = _read_champions(directory)
    champions = {row["root_id"]: row for row in champions_rows}
    if len(champions) != len(champions_rows):
        raise ValueError(f"Champion records contain duplicate roots: {directory}")
    identities = []
    counts = Counter()
    state_total = 0
    board_responses = 0
    max_trace = 0
    max_teacher_width = 0
    presentations_seen = Counter()
    expected_presentations = {}
    suffixes = {}
    roots_seen = set()
    checked_champions = set()
    for index, row in enumerate(rows):
        root_id = row.get("root_id")
        presentation = row.get("presentation")
        if not isinstance(root_id, str) or presentation not in presentations:
            raise ValueError(f"Release row identity is invalid: {directory}")
        teacher_width = row.get("teacher_width")
        depth = row.get("depth")
        if not _is_int(teacher_width, 1) or not _is_int(depth, 1):
            raise ValueError(f"Release teacher settings are invalid: {root_id}")
        if teacher_width != sample_width(recipe, seed, root_id, default_width):
            raise ValueError(f"Release teacher width differs: {root_id}")
        if depth != sample_depth(recipe, seed, root_id, default_depth):
            raise ValueError(f"Release depth differs: {root_id}")
        if root_id not in expected_presentations:
            expected_presentations[root_id] = sample_presentation(recipe, seed, root_id)
        expected_presentation = expected_presentations[root_id]
        if expected_presentation is not None and presentation != expected_presentation:
            raise ValueError(f"Release presentation differs from the recipe: {root_id}")
        champion = champions.get(root_id)
        if champion is None:
            raise ValueError(f"Release row has no champion record: {root_id}")
        if expected_roots is not None:
            expected = expected_roots.get(root_id)
            if expected is None:
                raise ValueError(f"Release contains an unselected root: {root_id}")
            for name in (*ROOT_ROW_FIELDS, "teacher_width", "depth"):
                if name in expected and row.get(name) != expected.get(name):
                    raise ValueError(
                        f"Release root {name} differs from its selection: {root_id}"
                    )
        board = chess.Board(row["fen"])
        if board.turn != (row["turn"] == "white"):
            raise ValueError(f"Root turn disagrees with FEN: {root_id}")
        effective_width = min(teacher_width, board.legal_moves.count())
        max_teacher_width = max(max_teacher_width, teacher_width)
        repetition = row["repetition"]
        if not _is_int(repetition, 1) or repetition > 4:
            raise ValueError(f"Root repetition is out of range: {root_id}")
        if root_id not in checked_champions:
            if len(champion.get("rounds", [])) != depth:
                raise ValueError(f"Champion record has the wrong depth: {root_id}")
            for entry in champion["rounds"]:
                _check_champion_round(root_id, entry, board, value_mate, max_ply)
            checked_champions.add(root_id)
        encoded = encode_board(row["fen"], repetition)
        board_tokens = arrays["board_tokens"][index]
        if not np.array_equal(board_tokens[: len(encoded)], encoded) or not np.all(
            board_tokens[len(encoded) :] == PAD
        ):
            raise ValueError(f"Board tokens disagree with FEN: {root_id}")

        length = int(arrays["trace_len"][index])
        tokens = arrays["trace_tokens"][index, :length]
        if not np.all(arrays["trace_tokens"][index, length:] == PAD) or not np.all(
            arrays["trace_roles"][index, length:] == 0
        ):
            raise ValueError(f"Trace padding is invalid: {root_id}")
        program = replay(
            tokens,
            row["fen"],
            repetition,
            teacher_width=effective_width,
        )
        if not np.array_equal(program.roles, arrays["trace_roles"][index, :length]):
            raise ValueError(f"Trace roles disagree with replay: {root_id}")
        max_trace = max(max_trace, length)

        board_spans = [
            (step.start, step.end)
            for step in program.steps
            if step.operation == "board"
        ]
        board_responses += len(board_spans)
        no_tool_tokens = list(tokens)
        for start, end in reversed(board_spans):
            del no_tool_tokens[start:end]
        no_tool = replay(
            no_tool_tokens,
            row["fen"],
            repetition,
            teacher_width=effective_width,
        )
        expected_boundaries = []
        operation = 0
        previous_no_tool_end = 0
        for step in program.steps:
            if step.operation == "board":
                expected_boundaries.append((step.end, previous_no_tool_end))
                continue
            no_tool_step = no_tool.steps[operation]
            if (
                step.position != no_tool_step.position
                or step.uci != no_tool_step.uci
                or step.evaluation != no_tool_step.evaluation
                or step.mate != no_tool_step.mate
            ):
                raise ValueError(f"Tool and no-tool traces differ: {root_id}")
            previous_no_tool_end = no_tool_step.end
            if step.operation not in FINAL_SUFFIX_OPS:
                expected_boundaries.append((step.end, no_tool_step.end))
            operation += 1
        if operation != len(no_tool.steps):
            raise ValueError(f"Tool and no-tool operations differ: {root_id}")
        expected_boundaries.sort()

        count = int(arrays["state_count"][index])
        if count != len(expected_boundaries):
            raise ValueError(f"Truncation-state count is invalid: {root_id}")
        actual_boundaries = list(
            zip(
                map(int, arrays["state_with_board"][index, :count]),
                map(int, arrays["state_without_board"][index, :count]),
            )
        )
        if actual_boundaries != expected_boundaries:
            raise ValueError(f"Truncation boundaries are invalid: {root_id}")
        for name in STATE_ARRAYS:
            if not np.all(arrays[name][index, count:] == -1):
                raise ValueError(f"Truncation-state padding is invalid: {root_id}")
        expected_states = _expected_states(
            program, tokens, champion, board.turn == chess.BLACK
        )
        if len(expected_states) != count:
            raise ValueError(f"Truncation-state count is invalid: {root_id}")
        for position, expected_state in enumerate(expected_states):
            actual = (
                int(arrays["state_best"][index, position]),
                int(arrays["state_value"][index, position]),
                int(arrays["state_basis"][index, position]),
            )
            if actual != expected_state:
                raise ValueError(f"Truncation state is invalid: {root_id}")

        final_step = next(step for step in program.steps if step.operation == "final")
        final_index = final_step.start
        suffix = tuple(int(token) for token in tokens[final_index:])
        if root_id in suffixes and suffixes[root_id] != suffix:
            raise ValueError(f"Presentations disagree on the FINAL suffix: {root_id}")
        suffixes[root_id] = suffix
        if row.get("thinking_tokens") != final_index:
            raise ValueError(f"Thinking-token count is invalid: {root_id}")
        rounds = sum(1 for token in tokens[:final_index] if token == DEPTH)
        if row.get("rounds") != rounds:
            raise ValueError(f"Row round count is invalid: {root_id}")
        final_move = program.final_move
        if row.get("final_move") != final_move:
            raise ValueError(f"FINAL move differs from the trace: {root_id}")
        if row.get("claim_draw") != (final_move == CLAIM_DRAW_UCI):
            raise ValueError(f"Draw-claim flag differs from the trace: {root_id}")
        value_token = int(tokens[final_index + 2])
        if row.get("final_value_bin") != value_token - VALUE:
            raise ValueError(f"FINAL value bin differs from the trace: {root_id}")
        mate_distance = None
        mate_index = final_index + 3
        if mate_index < len(tokens) and MATE_WIN <= int(tokens[mate_index]) < PAD:
            token = int(tokens[mate_index])
            mate_distance = (
                -(token - MATE_LOSS) if token >= MATE_LOSS else token - MATE_WIN
            )
        if row.get("mate_distance") != mate_distance:
            raise ValueError(f"FINAL mate distance differs from the trace: {root_id}")
        deepest = champion["rounds"][-1]
        if deepest["round"] != champion["rounds"].__len__():
            raise ValueError(f"Champion rounds are not contiguous: {root_id}")
        if final_move == CLAIM_DRAW_UCI:
            claimable = repetition >= 3 or board.halfmove_clock >= 100
            if not claimable or deepest["raw"] > DRAW_CLAIM_BOUND:
                raise ValueError(f"Draw claim is not justified: {root_id}")
        elif final_move != deepest["move"]:
            raise ValueError(f"FINAL move differs from the deepest champion: {root_id}")
        if deepest["mate"] != mate_distance:
            raise ValueError(
                f"FINAL mate distance differs from the champion: {root_id}"
            )
        if deepest["wdl"] is None:
            expected_bin = 19 if deepest["mate"] > 0 else 0
        else:
            expected_bin = value_bin((deepest["wdl"][0] - deepest["wdl"][2]) / 1000)
        if row.get("final_value_bin") != expected_bin:
            raise ValueError(f"FINAL value bin differs from the champion: {root_id}")
        pv_start = final_index + (4 if mate_distance is not None else 3)
        pv_tokens = [int(token) for token in tokens[pv_start:-1]]
        mirrored = board.turn == chess.BLACK
        actual_pv = [
            reflect_uci(IDX_TO_UCI_MOVE[token - MOVE])
            if mirrored
            else IDX_TO_UCI_MOVE[token - MOVE]
            for token in pv_tokens
        ]
        expected_pv = [] if final_move == CLAIM_DRAW_UCI else deepest["pv"][1:]
        if actual_pv != expected_pv:
            raise ValueError(f"FINAL variation differs from the champion: {root_id}")

        identities.append((root_id, presentation))
        counts[row["split"]] += 1
        counts[presentation] += 1
        state_total += count
        presentations_seen[root_id] += 1
        roots_seen.add(root_id)
    if roots_seen != set(champions):
        raise ValueError(f"Shard rows do not cover its champion records: {directory}")
    for root_id, count in presentations_seen.items():
        wanted = (
            1 if expected_presentations[root_id] is not None else len(PRESENTATIONS)
        )
        if count != wanted:
            raise ValueError(f"A shard root has the wrong row count: {directory}")
    for champion in champions.values():
        rounds = champion["rounds"]
        if [entry["round"] for entry in rounds] != list(range(1, len(rounds) + 1)):
            raise ValueError(f"Champion rounds are not contiguous: {directory}")
    return {
        "identities": identities,
        "counts": counts,
        "states": state_total,
        "board_responses": board_responses,
        "max_trace": max_trace,
        "max_teacher_width": max_teacher_width,
    }


_VALIDATION_CONTEXT = None


def _init_validation_worker(*context):
    global _VALIDATION_CONTEXT
    _VALIDATION_CONTEXT = context


def _validate_shard(shard):
    directory, *arguments = _VALIDATION_CONTEXT
    return _validate_shard_rows(directory, shard, *arguments)


def validate_release(directory, roots=None, workers=8):
    """Validate every file and replay every row of a complete sharded release."""
    if workers < 1:
        raise ValueError("Validation workers must be positive")
    directory = Path(directory)
    release = load_manifest(directory / "release.json")
    if not release:
        raise FileNotFoundError(directory / "release.json")
    expected_metadata = {
        "format_version": DATA_FORMAT_VERSION,
        "grammar_version": TRACE_GRAMMAR_VERSION,
        "tokenizer_version": CHESS_TOKENIZER_VERSION,
        "role_count": ROLE_COUNT,
    }
    for key, value in expected_metadata.items():
        if release.get(key) != value:
            raise ValueError(f"Release metadata {key} is unsupported")
    complete = release["complete_root_ids"]
    metadata = validate_release_identity(release)
    teacher = metadata["teacher"]
    search = metadata["search"]
    recipe = parse_recipe(metadata["recipe"]) if "recipe" in metadata else None
    seed = metadata["root_selection"]["seed"]
    presentations = (
        tuple(name for name, _ in recipe.presentations)
        if recipe is not None and recipe.presentations is not None
        else PRESENTATIONS
    )
    expected_shard_metadata = {**expected_metadata, **shard_configuration(metadata)}
    expected = None
    if roots is not None:
        root_records = list(roots)
        expected = {row["root_id"]: row for row in root_records}
        if len(expected) != len(root_records):
            raise ValueError("External root records contain duplicate root ids")
    seen = set()
    seen_roots = set()
    counts = Counter()
    states = 0
    board_responses = 0
    max_trace = 0
    max_teacher_width = 0

    context = (
        presentations,
        search["teacher_width"],
        search["depth"],
        recipe,
        seed,
        teacher["value_mate"],
        teacher["max_ply"],
        expected_shard_metadata,
        expected,
    )
    with ExitStack() as stack:
        if workers == 1:
            results = (
                _validate_shard_rows(directory, shard, *context)
                for shard in release["shards"]
            )
        else:
            pool = stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_validation_worker,
                    initargs=(directory, *context),
                )
            )
            results = pool.map(_validate_shard, release["shards"])
        for result in results:
            for root_id, presentation in result["identities"]:
                key = (root_id, presentation)
                if key in seen:
                    raise ValueError(f"Duplicate release row: {key}")
                seen.add(key)
                seen_roots.add(root_id)
            counts.update(result["counts"])
            states += result["states"]
            board_responses += result["board_responses"]
            max_trace = max(max_trace, result["max_trace"])
            max_teacher_width = max(max_teacher_width, result["max_teacher_width"])
    expected_schema = SIDECAR_SCHEMA_WIDE if max_teacher_width > 1 else SIDECAR_SCHEMA
    if teacher["schema_version"] != expected_schema:
        raise ValueError("Release teacher schema differs from its rows")
    if seen_roots != set(complete) or len(seen) != len(complete):
        raise ValueError("Release rows do not match its roots")
    if expected is not None:
        wanted = {
            (root_id, sample_presentation(recipe, seed, root_id))
            for root_id in expected
        }
        if seen != wanted or set(complete) != set(expected):
            raise ValueError("Release does not contain exactly the selected roots")
    result = {
        "ok": True,
        "roots": len(complete),
        "rows": len(seen),
        "states": states,
        "board_responses": board_responses,
        "max_trace": max_trace,
        "counts": dict(counts),
        "shards": len(release["shards"]),
    }
    write_manifest(directory / "validation.json", result)
    return result


def sample_root_settings(roots, config, seed):
    """Derive per-root settings from the active recipe and stable root ids."""
    sampled = []
    for root in roots:
        result = dict(root)
        result["teacher_width"] = sample_width(
            config.sampling, seed, root["root_id"], config.search.teacher_width
        )
        result["depth"] = sample_depth(
            config.sampling, seed, root["root_id"], config.search.depth
        )
        result["presentation"] = (
            sample_presentation(config.sampling, seed, root["root_id"]) or "full"
        )
        sampled.append(result)
    return sampled


def _engine_lines(engine_path, root, config):
    from tempfile import TemporaryDirectory

    width, depth = _root_teacher(root, config)
    with TemporaryDirectory(prefix="pondera-sidecar-") as directory:
        trace_path = Path(directory) / "trace.jsonl"
        bestmove = _run_engine(
            engine_path, root["fen"], config, trace_path, width, depth
        )
        if not trace_path.is_file():
            raise CollectionError(f"Engine wrote no sidecar for root {root['root_id']}")
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    return lines, bestmove


def _process_root(root, config, lines, bestmove=None, engine_name=None, teacher=None):
    """Run the complete single-root contract without any intermediate files."""
    width, _ = _root_teacher(root, config)
    sidecar = parse_sidecar(
        lines,
        teacher_width=width,
        source=f"root {root['root_id']}",
    )
    _check_sidecar(root, sidecar, config, engine_name or sidecar.engine, teacher)
    if bestmove is not None and sidecar.search_end["bestmove"] != bestmove:
        raise CollectionError(
            f"bestmove differs from the sidecar for {root['root_id']}"
        )
    synthesized = synthesize_root(sidecar, root, config)
    presentation = root.get("presentation") or "full"
    row = synthesized.rows[presentation]
    program = replay(
        row.tokens,
        root["fen"],
        int(root["repetition"]),
        teacher_width=sidecar.width,
    )
    if tuple(program.roles) != tuple(row.roles):
        raise SynthesisError(f"replayed roles differ for {root['root_id']}")
    return synthesized


def _provider_lines(provider, root):
    value = provider(root) if callable(provider) else provider[root["root_id"]]
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
        return value
    if isinstance(value, (str, bytes)):
        return str(value).splitlines(), None
    return list(value), None


def _direct_worker(root):
    lines, bestmove = _engine_lines(_DIRECT_ENGINE, root, _DIRECT_CONFIG)
    return _process_root(
        root,
        _DIRECT_CONFIG,
        lines,
        bestmove=bestmove,
        engine_name=_DIRECT_NAME,
        teacher=_DIRECT_TEACHER,
    )


def _init_direct_worker(config, engine_path, engine_name, teacher):
    global _DIRECT_CONFIG, _DIRECT_ENGINE, _DIRECT_NAME, _DIRECT_TEACHER
    _DIRECT_CONFIG = config
    _DIRECT_ENGINE = engine_path
    _DIRECT_NAME = engine_name
    _DIRECT_TEACHER = teacher


def _identity_for(config, roots, engine_name="stockfish"):
    root_path = Path(config.paths.roots)
    return {
        "teacher": {
            "schema_version": (
                SIDECAR_SCHEMA_WIDE
                if any(
                    root.get("teacher_width", config.search.teacher_width) > 1
                    for root in roots
                )
                else SIDECAR_SCHEMA
            ),
            "value_mate": 32000,
            "max_ply": 246,
        },
        "engine": {"name": engine_name},
        "search": config.search_dict(),
        "board": config.board_dict(),
        "presentation": config.presentation_dict(),
        "synthesis": config.synthesis_dict(),
        "root_selection": {
            "source": str(root_path),
            "algorithm": "trace.toml",
            "seed": config.seed,
            "requested": len(roots),
            "counts": {
                split: sum(root["split"] == split for root in roots)
                for split in ("train", "val")
            },
            "source_counts": {
                split: sum(root["split"] == split for root in roots)
                for split in ("train", "val")
            },
        },
        "recipe": config.sampling.identity(),
    }


def _ensure_root_pool(config):
    """Load or directly build the cached root pool required by the config."""
    from pondera.data.roots import build_pool, ensure_archive, read_roots

    path = Path(config.paths.roots)
    parameters = {
        "train": config.roots.train,
        "val": config.roots.val,
        "claimable_share": config.roots.claimable_share,
    }
    manifest = (
        load_manifest(path.parent / "pool.manifest.json") if path.is_file() else {}
    )
    expected_source = Path(config.paths.archive)
    cache_matches = (
        path.is_file()
        and manifest.get("kind") == "elite-root-pool"
        and manifest.get("seed") == config.seed
        and manifest.get("parameters") == parameters
        and manifest.get("source", {}).get("file") == expected_source.name
    )
    if not path.is_file() or (manifest and not cache_matches):
        archive = expected_source
        if not archive.is_file():
            archive = ensure_archive(archive.parent)
        build_pool(
            archive,
            path.parent,
            config.roots.train,
            config.roots.val,
            config.roots.claimable_share,
            config.seed,
        )
        generated = path.parent / "roots.jsonl"
        if generated != path:
            generated.replace(path)
    raw_roots = read_roots(path)
    roots = sample_root_settings(raw_roots, config, config.seed)
    if roots != raw_roots:
        _write_jsonl(path, roots)
    return roots


@contextmanager
def _release_lock(output_dir):
    with (output_dir / ".release.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def produce_shards(
    roots,
    output_dir,
    config,
    sidecar_for_root=None,
    identity=None,
    shard_size=None,
    workers=None,
    start_shard=0,
    end_shard=None,
):
    """Produce shards directly from roots, with optional fixture sidecars.

    ``sidecar_for_root`` is a callable/mapping used by tests and offline
    converters. When it is omitted, each worker runs a fresh instrumented
    engine and the sidecar exists only in a temporary directory.
    """
    roots = sample_root_settings(list(roots), config, config.seed)
    if not roots:
        raise ValueError("A trace release needs at least one root")
    if len({root["root_id"] for root in roots}) != len(roots):
        raise ValueError("A trace release cannot contain duplicate root ids")
    shard_size = config.release.shard_size if shard_size is None else shard_size
    workers = config.release.workers if workers is None else workers
    if shard_size < 1 or workers < 1:
        raise ValueError("Shard size and workers must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_path = None
    engine_name = "stockfish"
    teacher = None
    if sidecar_for_root is None:
        engine_path = _engine_path(config)
        engine_name = "Stockfish 17.1"
        teacher = (32000, 246)
    identity = identity or _identity_for(config, roots, engine_name=engine_name)
    release_metadata = _release_metadata(identity)
    shard_metadata = {
        key: value
        for key, value in release_metadata.items()
        if key not in RELEASE_ONLY_METADATA
    }
    presentations = {
        root["root_id"]: (root.get("presentation") or "full",) for root in roots
    }
    release_path = output_dir / "release.json"
    previous = load_manifest(release_path)
    if previous and previous.get("metadata") != identity:
        raise ValueError("The existing release belongs to a different configuration")
    shards_by_index = {
        int(shard["path"].split("-")[-1]): shard for shard in previous.get("shards", [])
    }
    groups = [
        (index, roots[start : start + shard_size])
        for index, start in enumerate(range(0, len(roots), shard_size))
    ]
    for index, group in groups:
        if index < start_shard or (end_shard is not None and index >= end_shard):
            continue
        if index in shards_by_index:
            continue
        final = output_dir / f"shard-{index:06d}"
        if final.exists():
            shard = _recover_shard(
                output_dir, index, group, shard_metadata, presentations
            )
        else:
            if sidecar_for_root is not None:
                records = []
                for root in group:
                    lines, bestmove = _provider_lines(sidecar_for_root, root)
                    records.append(
                        _process_root(root, config, lines, bestmove=bestmove)
                    )
            elif workers == 1:
                records = []
                for root in group:
                    lines, bestmove = _engine_lines(engine_path, root, config)
                    records.append(
                        _process_root(
                            root,
                            config,
                            lines,
                            bestmove=bestmove,
                            engine_name=engine_name,
                            teacher=teacher,
                        )
                    )
            else:
                with ProcessPoolExecutor(
                    max_workers=min(workers, len(group)),
                    initializer=_init_direct_worker,
                    initargs=(config, engine_path, engine_name, teacher),
                ) as pool:
                    records = list(pool.map(_direct_worker, group))
            shard = _publish_shard(
                output_dir, index, records, shard_metadata, presentations
            )
        with _release_lock(output_dir):
            current = load_manifest(release_path)
            if current and current.get("metadata") != identity:
                raise ValueError(
                    "The existing release belongs to a different configuration"
                )
            shards_by_index = {
                int(item["path"].split("-")[-1]): item
                for item in current.get("shards", [])
            }
            shards_by_index[index] = shard
            complete = set(current.get("complete_root_ids", []))
            complete.update(root["root_id"] for root in group)
            write_manifest(
                release_path,
                {
                    **release_metadata,
                    "metadata": identity,
                    "shards": [shards_by_index[key] for key in sorted(shards_by_index)],
                    "complete_root_ids": sorted(complete),
                },
            )
    with _release_lock(output_dir):
        current = load_manifest(release_path)
        complete = set(current.get("complete_root_ids", []))
        profile = RootProfile(presentations)
        for shard in current.get("shards", []):
            _profile_shard(output_dir, shard, shard_metadata, profile)
        statistics = profile.result() if profile.roots else None
        if statistics is not None:
            statistics["complete"] = len(complete) == len(roots)
            write_manifest(output_dir / "statistics.json", statistics)
    return {
        "shards": current.get("shards", []),
        "roots": len(complete),
        "pending": len(roots) - len(complete),
        "release": str(release_path),
        "statistics": statistics,
    }


def run_pipeline(config, start_shard=0, end_shard=None, workers=None):
    roots = _ensure_root_pool(config)
    return produce_shards(
        roots,
        config.paths.output,
        config,
        workers=workers,
        start_shard=start_shard,
        end_shard=end_shard,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), type=Path)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    result = run_pipeline(
        load_config(args.config),
        start_shard=args.start_shard,
        end_shard=args.end_shard,
        workers=args.workers,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "statistics"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
