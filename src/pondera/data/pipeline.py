"""Offline trace production: configuration, engine execution, shards, and checks.

The public entry point is ``run_pipeline``. It reads a prepared root pool,
then runs engine -> sidecar decoding -> synthesis for each root, checkpointing
every collected root beside the release so interrupted runs recollect only
missing roots, and publishes complete shards atomically. Exhaustive replay is
an explicit diagnostic, outside normal production.
"""

import fcntl
import gc
import hashlib
import json
import os
import pickle
import random
import select
import shutil
import signal
import statistics
import subprocess
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from multiprocessing import get_context
from multiprocessing.util import Finalize
from pathlib import Path
from uuid import uuid4

import chess
import numpy as np

from pondera.chess.codec import cap_mate_distance, value_bin
from pondera.chess.symmetry import reflect_uci
from pondera.chess.vocabulary import (
    CHESS_TOKENIZER_VERSION,
    FINAL,
    IDX_TO_UCI_MOVE,
    MATE_LOSS,
    MATE_WIN,
    MOVE,
    PAD,
    VALUE,
)
from pondera.data.config import PRESENTATION_NAMES, parse_recipe
from pondera.data.dataset import (
    DATA_ARRAYS,
    DATA_FORMAT_VERSION,
    load_manifest,
    load_trace_arrays,
    read_root_rows,
    validate_trace_arrays,
    write_manifest,
    write_trace_rows,
)
from pondera.data.trace import (
    CLAIM_DRAW_UCI,
    DRAW_CLAIM_BOUND,
    PRESENTATIONS,
    ROLE_COUNT,
    SIDECAR_SCHEMA,
    SIDECAR_SCHEMA_WIDE,
    TRACE_GRAMMAR_VERSION,
    decode_sidecar,
    mate_distance_moves,
    parse_sidecar,
    replay,
    synthesize_root,
    validate_root,
)


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


def sample_search(sampling, seed, root_id):
    return sample_distribution(
        random.Random(f"{seed}:{root_id}:search"), sampling.joint
    )


def sample_presentation(sampling, seed, root_id):
    return sample_distribution(
        random.Random(f"{seed}:{root_id}:presentation"), sampling.presentations
    )


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

    def add_stored(self, row, stats, token_count, thinking_tokens):
        self._add(
            row["root_id"],
            row["split"],
            stats,
            [(row["presentation"], token_count, thinking_tokens)],
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


class EngineTimeoutError(CollectionError):
    """The engine exceeded the per-root deadline; the root is dropped."""


class EngineStartupError(CollectionError):
    """The worker cannot start an engine; no root attempt was made."""


# A root that keeps failing stops being retried but is never dropped silently:
# its shard stays unpublished and the run fails loudly once everything else
# is done. The run-level bound stops a systemic failure (e.g. a full disk)
# from burning one retry budget on every remaining root.
MAX_ROOT_ATTEMPTS = 3
MAX_RUN_FAILURES = 64


def _root_teacher(root):
    width = root["teacher_width"]
    depth = root["depth"]
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise CollectionError(f"Root {root['root_id']} has an invalid teacher width")
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        raise CollectionError(f"Root {root['root_id']} has an invalid depth")
    return width, depth


def _engine_path(engine):
    path = Path(engine)
    if path.is_file():
        return path.resolve()
    resolved = shutil.which(engine)
    if resolved is None:
        raise FileNotFoundError(engine)
    return Path(resolved).resolve()


def _start_engine(engine_path, config):
    """Start an engine, separating worker failures from root failures."""
    try:
        return _Engine(
            engine_path,
            config.search.hash_mb,
            config.search.timeout_seconds,
            config.search.syzygy_path,
            config.search.syzygy_probe_limit,
        )
    except (CollectionError, OSError) as error:
        raise EngineStartupError(f"engine failed to start: {error}") from error


def _check_sidecar(root, sidecar, config, engine_name, teacher=None):
    width, depth = _root_teacher(root)
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


class _Engine:
    """One worker's UCI process, reset to fresh search state for each root."""

    def __init__(
        self, path, hash_mb, timeout=3600, syzygy_path=None, syzygy_probe_limit=7
    ):
        if syzygy_path is not None and (
            not Path(syzygy_path).is_dir() or not any(Path(syzygy_path).glob("*.rtbw"))
        ):
            # Probing must fail loudly here; an unusable directory would
            # silently produce ordinary searches under a probing identity.
            raise CollectionError(f"Syzygy tablebases are unusable: {syzygy_path}")
        self.process = subprocess.Popen(
            [str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
        )
        self.buffer = b""
        self.timeout = timeout
        self.closed = False
        try:
            deadline = time.monotonic() + timeout
            self._send("uci")
            _, banner = self._read_until("uciok", deadline)
            if not any(line.startswith("option name TraceLog ") for line in banner):
                raise CollectionError(f"Engine does not expose TraceLog: {path}")
            self._send("setoption name Threads value 1")
            self._send(f"setoption name Hash value {hash_mb}")
            if syzygy_path is not None:
                self._send(f"setoption name SyzygyPath value {syzygy_path}")
                self._send(
                    f"setoption name SyzygyProbeLimit value {syzygy_probe_limit}"
                )
        except BaseException:
            self.close(force=True)
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _send(self, line):
        try:
            self.process.stdin.write((line + "\n").encode())
            self.process.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as error:
            raise CollectionError("engine process is gone") from error

    def _read_line(self, deadline):
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngineTimeoutError("engine timed out")
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                continue
            chunk = self.process.stdout.read(65536)
            if not chunk:
                raise CollectionError("engine closed its output")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.decode(errors="replace").strip()

    def _read_until(self, prefix, deadline):
        seen = []
        while True:
            line = self._read_line(deadline)
            seen.append(line)
            if line.startswith(prefix):
                return line, seen

    def search(self, fen, trace_path, width, depth):
        deadline = time.monotonic() + self.timeout
        self._send("setoption name TraceLog value ")
        # Stockfish 17.1 clears the TT, search histories and NNUE refresh cache.
        self._send("ucinewgame")
        self._send(f"setoption name MultiPV value {width}")
        self._send(f"setoption name TraceLog value {trace_path}")
        self._send("isready")
        self._read_until("readyok", deadline)
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        bestmove, _ = self._read_until("bestmove", deadline)
        parts = bestmove.split()
        if len(parts) < 2:
            raise CollectionError(f"engine returned no bestmove: {bestmove!r}")
        return parts[1]

    def close(self, *, force=False):
        if self.closed:
            return
        if force:
            self.process.kill()
            self.process.wait()
        else:
            try:
                self._send("quit")
            except CollectionError:
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.closed = True


RELEASE_ONLY_METADATA = frozenset({"root_selection"})
ROOT_ROW_FIELDS = ("split", "fen", "repetition", "turn", "source")
CHAMPION_FIELDS = ("round", "depth", "move", "raw", "wdl", "mate", "value_token", "pv")


def _write_jsonl(path, rows):
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _root_row(root, synthesized, presentation):
    row = synthesized.rows[presentation]
    payload = {
        "root_id": root["root_id"],
        "presentation": presentation,
        "teacher_width": synthesized.stats["teacher_width"],
        "teacher_depth": synthesized.stats["depth"],
        "trace_text": row.trace_text,
        "final_value": (
            synthesized.round_audits[-1].wdl[0] - synthesized.round_audits[-1].wdl[2]
        )
        / 1000
        if synthesized.round_audits[-1].wdl is not None
        else None,
        "trace_len": len(row.tokens),
        "final_move": row.final_move,
        "mate_distance": row.mate_distance,
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
    rows, champion_rows = [], []
    for synthesized in records:
        root = synthesized.root
        for presentation in presentations[root["root_id"]]:
            row = synthesized.rows[presentation]
            payload = _root_row(root, synthesized, presentation)
            payload["trace_token_ids"] = row.tokens
            for name, field in (("state_best", "best"), ("state_value", "value")):
                payload[name] = [
                    getattr(boundary, field) for boundary in row.boundaries
                ]
            rows.append(payload)
        champion_rows.append(_champion_row(root, synthesized))
    # An all-dropped shard legitimately has no rows; release-level checks
    # (rows == selected minus dropped) catch a wrongly empty one.
    write_trace_rows(directory, rows)
    _write_jsonl(directory / "champions.jsonl", champion_rows)
    payload = {**metadata, "rows": len(rows)}
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
    _, rows = _read_shard_metadata(output, published, metadata)
    if {(row["root_id"], row["presentation"]) for row in rows} != _expected_root_rows(
        roots, presentations
    ):
        raise FileExistsError(f"Unregistered shard has different roots: {final}")
    expected = {root["root_id"]: root for root in roots}
    for row in rows:
        root = expected[row["root_id"]]
        for name in (*ROOT_ROW_FIELDS, "teacher_width", "depth"):
            expected_value = root.get(name)
            if (
                name in root
                and row.get("teacher_depth" if name == "depth" else name)
                != expected_value
            ):
                raise FileExistsError(
                    f"Unregistered shard root {name} differs: {final}"
                )
    return published


def _check_shard_layout(output, shard, roots):
    """Reject a registered shard that no longer matches the current layout."""
    stored = read_root_rows(Path(output) / shard["path"])
    if {row["root_id"] for row in stored} != {root["root_id"] for root in roots}:
        raise ValueError(
            f"Registered shard {shard['path']} covers different roots; "
            "the release was produced with a different shard layout"
        )
    expected = {root["root_id"]: root for root in roots}
    for row in stored:
        root = expected[row["root_id"]]
        for name in ROOT_ROW_FIELDS:
            if name in root and row.get(name) != root[name]:
                raise ValueError(f"Registered root {name} differs: {row['root_id']}")


def _profile_shard(output, shard, profile):
    directory = Path(output) / shard["path"]
    rows = read_root_rows(directory)
    tokens = load_trace_arrays(directory, ("trace_token_ids",))["trace_token_ids"]
    champions = {row["root_id"]: row for row in _read_champions(directory)}
    for index, row in enumerate(rows):
        final = np.flatnonzero(tokens[index] == FINAL)
        if len(final) != 1:
            raise ValueError(f"Stored trace has no unique FINAL: {row['root_id']}")
        profile.add_stored(
            row,
            champions[row["root_id"]]["stats"],
            row["trace_len"],
            int(final[0]),
        )


def _publish_shard(output, index, records, metadata, presentations):
    final = output / f"shard-{index:06d}"
    # A unique staging directory: two producers erroneously targeting the
    # same shard (the contract is disjoint ranges) cannot wreck each other's
    # staging area.
    temporary = output / f".shard-{index:06d}.{uuid4().hex}.tmp"
    # Dropped roots (engine timeouts) leave no record behind; a shard whose
    # roots all timed out publishes with zero rows.
    records = [record for record in records if record is not None]
    try:
        payload = _write_shard(temporary, records, metadata, presentations)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    if final.exists():
        shutil.rmtree(temporary)
        roots = [record.root for record in records]
        return _recover_shard(output, index, roots, metadata, presentations)
    os.replace(temporary, final)
    return {"path": final.name, "rows": payload["rows"]}


class _RootCache:
    """Per-root synthesized-record checkpoints beside a release's shards.

    Every successfully collected root is pickled under ``.roots`` so an
    interrupted run recollects only roots that never completed. Entries are
    tied to the release identity; an unreadable or root-mismatching entry
    silently degrades to recollection, since a cache may never corrupt a
    release. Failure records count attempts across runs so a permanently
    failing root stops being retried without ever being dropped silently.
    Dropped roots (engine timeouts) persist as markers so a resumed run
    never recollects them.
    """

    def __init__(self, output, identity):
        self.directory = output / ".roots"
        marker = self.directory / "identity.json"
        stored = None
        if marker.is_file():
            try:
                stored = load_manifest(marker)
            except (OSError, json.JSONDecodeError):
                stored = None
            if stored != identity:
                # An unknown or unreadable provenance voids the whole cache.
                shutil.rmtree(self.directory)
                stored = None
        self.directory.mkdir(parents=True, exist_ok=True)
        if stored is None:
            write_manifest(marker, identity)

    @staticmethod
    def _key(root):
        # Root ids are opaque strings (FEN-derived ids contain "/" and "|"),
        # never safe to interpolate into a path.
        return hashlib.sha256(root["root_id"].encode("utf-8")).hexdigest()

    def _shard(self, index):
        return self.directory / f"shard-{index:06d}"

    def load(self, index, root):
        path = self._shard(index) / f"{self._key(root)}.pkl"
        try:
            with path.open("rb") as handle:
                record = pickle.load(handle)
        except (
            OSError,
            EOFError,
            ValueError,
            TypeError,
            AttributeError,
            ImportError,
            pickle.UnpicklingError,
        ):
            return None
        if getattr(record, "root", None) != root:
            return None
        return record

    def store(self, index, root, record):
        shard = self._shard(index)
        key = self._key(root)
        shard.mkdir(parents=True, exist_ok=True)
        temporary = shard / f".{key}.{uuid4().hex}.tmp"
        try:
            with temporary.open("wb") as handle:
                pickle.dump(record, handle, protocol=5)
            temporary.replace(shard / f"{key}.pkl")
            (shard / f"{key}.error.json").unlink(missing_ok=True)
        finally:
            temporary.unlink(missing_ok=True)

    def failure(self, index, root):
        path = self._shard(index) / f"{self._key(root)}.error.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            not isinstance(record, dict)
            or record.get("root_id") != root["root_id"]
            or not _is_int(record.get("attempts"), 1)
            or not isinstance(record.get("error"), str)
        ):
            return None
        return record

    def record_failure(self, index, root, error):
        failure = self.failure(index, root)
        attempts = (failure["attempts"] if failure is not None else 0) + 1
        shard = self._shard(index)
        shard.mkdir(parents=True, exist_ok=True)
        key = self._key(root)
        path = shard / f"{key}.error.json"
        temporary = shard / f".{path.name}.{uuid4().hex}.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "root_id": root["root_id"],
                    "attempts": attempts,
                    "error": str(error),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return attempts

    def dropped(self, index, root):
        path = self._shard(index) / f"{self._key(root)}.dropped.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            not isinstance(record, dict)
            or record.get("root_id") != root["root_id"]
            or not isinstance(record.get("error"), str)
        ):
            return None
        return record

    def record_dropped(self, index, root, error):
        shard = self._shard(index)
        shard.mkdir(parents=True, exist_ok=True)
        key = self._key(root)
        path = shard / f"{key}.dropped.json"
        temporary = shard / f".{path.name}.{uuid4().hex}.tmp"
        temporary.write_text(
            json.dumps({"root_id": root["root_id"], "error": str(error)}) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def clear_shard(self, index):
        shutil.rmtree(self._shard(index), ignore_errors=True)


def _cached_or_capped(cache, index, root):
    """A root's checkpoint hit, its exhausted failure record, or its drop."""
    cached = cache.load(index, root)
    if cached is not None:
        return cached, None, False
    failure = cache.failure(index, root)
    if failure is not None and failure["attempts"] >= MAX_ROOT_ATTEMPTS:
        return None, failure["error"], False
    return None, None, cache.dropped(index, root) is not None


def _note_dropped(cache, index, root):
    cache.record_dropped(index, root, "engine timed out")
    print(
        f"shard {index:06d} root {root['root_id']}: dropped after engine timeout",
        flush=True,
    )


def _note_failure(cache, run, index, root, error):
    attempts = cache.record_failure(index, root, error)
    run["failures"] += 1
    print(
        f"shard {index:06d} root {root['root_id']}: collection failed "
        f"({attempts}/{MAX_ROOT_ATTEMPTS}): {error}",
        flush=True,
    )
    if run["failures"] > MAX_RUN_FAILURES:
        raise CollectionError(
            f"over {MAX_RUN_FAILURES} failed root attempts in one run; "
            "treating the failures as systemic"
        )
    return str(error)


def _release_metadata(identity):
    return {
        "format_version": DATA_FORMAT_VERSION,
        "grammar_version": TRACE_GRAMMAR_VERSION,
        "role_count": ROLE_COUNT,
        "tokenizer_version": CHESS_TOKENIZER_VERSION,
        **identity,
    }


FINAL_SUFFIX_OPS = frozenset({"final", "final_evaluate", "final_pv", "end"})
SEARCH_FIELDS = frozenset(
    {"hash_mb", "timeout_seconds", "syzygy_probe_limit", "syzygy_tables"}
)
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
        "recipe",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"Release identity is missing {sorted(missing)[0]}")
    recipe = parse_recipe(metadata["recipe"])

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
    if not _is_int(search["hash_mb"], 1):
        raise ValueError("Release search limits are invalid")
    if not _is_int(search["syzygy_probe_limit"], 0) or search["syzygy_probe_limit"] > 7:
        raise ValueError("Release tablebase probe limit is invalid")

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
    if any(
        presentation[name] != dict(recipe.presentations).get(name, 0.0)
        for name in PRESENTATION_NAMES
    ):
        raise ValueError("Release presentation fractions differ from the recipe")
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
    rows = sum(shard["rows"] for shard in release["shards"])
    if rows < 1:
        raise ValueError("A release needs completed rows")
    # The root-selection identity describes the selected roots, which are the
    # published rows plus the dropped ones.
    dropped = release.get("dropped", [])
    return _release_identity_contract(release, rows + len(dropped))


def _read_shard_metadata(release_dir, shard, expected_metadata=None):
    directory = Path(release_dir) / shard["path"]
    manifest = load_manifest(directory / "manifest.json")
    if expected_metadata is not None:
        for key, value in shard_configuration(expected_metadata).items():
            if manifest.get(key) != value:
                raise ValueError(f"Shard metadata {key} differs from its release")
    if manifest["rows"] != shard["rows"]:
        raise ValueError(f"Shard row count differs from its release: {directory}")
    roots = read_root_rows(directory)
    if len(roots) != manifest["rows"]:
        raise ValueError(f"Shard roots do not match its manifest: {directory}")
    return manifest, roots


def verify_shard_files(release_dir, shard, expected_metadata=None):
    """Diagnose a published shard's metadata and ragged array contract."""
    manifest, roots = _read_shard_metadata(release_dir, shard, expected_metadata)
    directory = Path(release_dir) / shard["path"]
    arrays = load_trace_arrays(directory, DATA_ARRAYS)
    validate_trace_arrays(arrays, rows=len(roots))
    return manifest, roots, arrays


def _read_champions(directory):
    return _read_jsonl(Path(directory) / "champions.jsonl")


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
        expected_mate = cap_mate_distance(
            mate_distance_moves(entry["raw"], value_mate, max_ply)
        )
        if expected_mate != mate:
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
        depth = row.get("teacher_depth")
        if not _is_int(teacher_width, 1) or not _is_int(depth, 1):
            raise ValueError(f"Release teacher settings are invalid: {root_id}")
        board = chess.Board(row["fen"])
        legal_width = board.legal_moves.count()
        width, expected_depth = sample_search(recipe, seed, root_id)
        expected_width = legal_width if width == "full" else width
        if teacher_width != expected_width:
            raise ValueError(f"Release teacher width differs: {root_id}")
        if depth != expected_depth:
            raise ValueError(f"Release depth differs: {root_id}")
        if root_id not in expected_presentations:
            expected_presentations[root_id] = sample_presentation(recipe, seed, root_id)
        expected_presentation = expected_presentations[root_id]
        if presentation != expected_presentation:
            raise ValueError(f"Release presentation differs from the recipe: {root_id}")
        champion = champions.get(root_id)
        if champion is None:
            raise ValueError(f"Release row has no champion record: {root_id}")
        if expected_roots is not None:
            expected = expected_roots.get(root_id)
            if expected is None:
                raise ValueError(f"Release contains an unselected root: {root_id}")
            for name in (*ROOT_ROW_FIELDS, "teacher_width", "depth"):
                expected_value = expected.get(name)
                if (
                    name in expected
                    and row.get("teacher_depth" if name == "depth" else name)
                    != expected_value
                ):
                    raise ValueError(
                        f"Release root {name} differs from its selection: {root_id}"
                    )
        if board.turn != (row["turn"] == "white"):
            raise ValueError(f"Root turn disagrees with FEN: {root_id}")
        effective_width = min(teacher_width, legal_width)
        max_teacher_width = max(max_teacher_width, teacher_width)
        repetition = row["repetition"]
        if not _is_int(repetition, 1) or repetition > 4:
            raise ValueError(f"Root repetition is out of range: {root_id}")
        deepest_wdl = champion["rounds"][-1]["wdl"]
        expected_value = (
            None
            if deepest_wdl is None
            else float(np.float32((deepest_wdl[0] - deepest_wdl[2]) / 1000))
        )
        if row["final_value"] != expected_value:
            raise ValueError(f"Final value differs from its teacher: {root_id}")
        if root_id not in checked_champions:
            if len(champion.get("rounds", [])) != depth:
                raise ValueError(f"Champion record has the wrong depth: {root_id}")
            for entry in champion["rounds"]:
                _check_champion_round(root_id, entry, board, value_mate, max_ply)
            checked_champions.add(root_id)

        length = int(arrays["trace_len"][index])
        tokens = arrays["trace_token_ids"][index]
        program = replay(
            tokens,
            row["fen"],
            repetition,
            teacher_width=effective_width,
        )
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
        expected_states = _expected_states(
            program, tokens, champion, board.turn == chess.BLACK
        )
        if len(expected_states) != count:
            raise ValueError(f"Truncation-state count is invalid: {root_id}")
        for position, expected_state in enumerate(expected_states):
            actual = (
                int(arrays["state_best"][index][position]),
                int(arrays["state_value"][index][position]),
            )
            if actual != expected_state[:2]:
                raise ValueError(f"Truncation state is invalid: {root_id}")

        final_step = next(step for step in program.steps if step.operation == "final")
        final_index = final_step.start
        suffix = tuple(int(token) for token in tokens[final_index:])
        if root_id in suffixes and suffixes[root_id] != suffix:
            raise ValueError(f"Presentations disagree on the FINAL suffix: {root_id}")
        suffixes[root_id] = suffix
        final_move = program.final_move
        if row.get("final_move") != final_move:
            raise ValueError(f"FINAL move differs from the trace: {root_id}")
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
        if count != 1:
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
    dropped = release.get("dropped", [])
    if not isinstance(dropped, list):
        raise ValueError("Release dropped list is invalid")
    dropped_ids = set()
    for entry in dropped:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"root_id", "shard"}
            or not isinstance(entry["root_id"], str)
            or not _is_int(entry["shard"], 0)
            or entry["root_id"] in dropped_ids
        ):
            raise ValueError("Release dropped entries are invalid")
        dropped_ids.add(entry["root_id"])
    expected_metadata = {
        "format_version": DATA_FORMAT_VERSION,
        "grammar_version": TRACE_GRAMMAR_VERSION,
        "tokenizer_version": CHESS_TOKENIZER_VERSION,
        "role_count": ROLE_COUNT,
    }
    for key, value in expected_metadata.items():
        if release.get(key) != value:
            raise ValueError(f"Release metadata {key} is unsupported")
    complete = sum(shard["rows"] for shard in release["shards"])
    metadata = validate_release_identity(release)
    teacher = metadata["teacher"]
    recipe = parse_recipe(metadata["recipe"])
    seed = metadata["root_selection"]["seed"]
    presentations = tuple(name for name, _ in recipe.presentations)
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
                    mp_context=get_context("spawn"),
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
    if teacher["schema_version"] == SIDECAR_SCHEMA and max_teacher_width > 1:
        raise ValueError("Release rows exceed its teacher schema")
    # A wide-schema release whose wide roots were all dropped is valid: the
    # schema is a property of the recipe, the rows a property of the survivors.
    if len(seen_roots) != complete or len(seen) != complete:
        raise ValueError("Release rows do not match its roots")
    if dropped_ids & seen_roots:
        raise ValueError("A dropped root appears in the release")
    if expected is not None:
        if not dropped_ids <= set(expected):
            raise ValueError("Release drops an unselected root")
        wanted = {
            (root_id, sample_presentation(recipe, seed, root_id))
            for root_id in expected
            if root_id not in dropped_ids
        }
        if seen != wanted:
            raise ValueError("Release does not contain exactly the selected roots")
    result = {
        "ok": True,
        "roots": complete,
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
        width, result["depth"] = sample_search(config.sampling, seed, root["root_id"])
        legal_width = chess.Board(root["fen"]).legal_moves.count()
        result["teacher_width"] = legal_width if width == "full" else width
        result["effective_width"] = min(result["teacher_width"], legal_width)
        result["presentation"] = sample_presentation(
            config.sampling, seed, root["root_id"]
        )
        sampled.append(result)
    return sampled


def _engine_sidecar(engine, root):
    from tempfile import TemporaryDirectory

    width, depth = _root_teacher(root)
    with TemporaryDirectory(prefix="pondera-sidecar-") as directory:
        trace_path = Path(directory) / "trace.jsonl"
        try:
            bestmove = engine.search(root["fen"], trace_path, width, depth)
        except EngineTimeoutError as error:
            raise EngineTimeoutError(f"{error} for root {root['root_id']}") from error
        except CollectionError as error:
            raise CollectionError(f"{error} for root {root['root_id']}") from error
        if not trace_path.is_file():
            raise CollectionError(f"Engine wrote no sidecar for root {root['root_id']}")
        with trace_path.open("rb") as handle:
            try:
                sidecar = decode_sidecar(handle)
            except ValueError as error:
                # A truncated or malformed sidecar is an engine-side failure,
                # so the root is isolated and retried like any lost search.
                # (SynthesisError and orjson.JSONDecodeError are ValueErrors.)
                raise CollectionError(
                    f"Engine wrote a malformed sidecar for root"
                    f" {root['root_id']}: {error}"
                ) from error
    return sidecar, bestmove


def _process_root(root, config, sidecar, bestmove=None, engine_name=None, teacher=None):
    """Synthesize one decoded producer record after checking search completion."""
    _check_sidecar(root, sidecar, config, engine_name or sidecar.engine, teacher)
    if bestmove is not None and sidecar.search_end["bestmove"] != bestmove:
        raise CollectionError(
            f"bestmove differs from the sidecar for {root['root_id']}"
        )
    return synthesize_root(sidecar, root, config)


def _provider_lines(provider, root):
    value = provider(root) if callable(provider) else provider[root["root_id"]]
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
        return value
    if isinstance(value, (str, bytes)):
        return str(value).splitlines(), None
    return list(value), None


def _collect_root(root, config, engine, engine_name, teacher, provider=None):
    """Build one row without cyclic-GC scans of temporary acyclic events."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        if provider is not None:
            lines, bestmove = _provider_lines(provider, root)
            sidecar = decode_sidecar(lines)
        else:
            sidecar, bestmove = _engine_sidecar(engine, root)
        try:
            result = _process_root(
                root,
                config,
                sidecar,
                bestmove=bestmove,
                engine_name=None if provider is not None else engine_name,
                teacher=teacher,
            )
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
            if provider is not None:
                raise
            # Diagnose producer data only on failure. Valid sidecars leave
            # synthesis bugs visible, without replaying every healthy search.
            try:
                parse_sidecar(
                    (json.dumps(event) for event in sidecar.events),
                    teacher_width=root["teacher_width"],
                )
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                raise CollectionError(
                    f"Engine wrote an invalid sidecar for root {root['root_id']}:"
                    f" {error}"
                ) from error
            raise
        # Release the event graph before automatic collection resumes.
        del sidecar
        return result
    finally:
        if enabled:
            gc.enable()


def _close_direct_engine():
    global _DIRECT_ENGINE
    if _DIRECT_ENGINE is not None:
        _DIRECT_ENGINE.close(force=True)
        _DIRECT_ENGINE = None


def _direct_worker(index, root):
    global _DIRECT_ENGINE
    if _DIRECT_STOPPING.is_set():
        raise InterruptedError("Curation stopped")
    # Startup failures abort the run before a root attempt is counted.
    if _DIRECT_ENGINE is None:
        _DIRECT_ENGINE = _start_engine(_DIRECT_ENGINE_PATH, _DIRECT_CONFIG)
    try:
        record = _collect_root(
            root, _DIRECT_CONFIG, _DIRECT_ENGINE, _DIRECT_NAME, _DIRECT_TEACHER
        )
        _DIRECT_CACHE.store(index, root, record)
    except BaseException:
        _close_direct_engine()
        raise


def _interrupt_direct_worker(signum, frame):
    # The executor catches task exceptions; leave subsequent interruption
    # disabled while the task unwinds and reaps its engine process.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise InterruptedError("Curation stopped")


def _init_direct_worker(
    config, engine_path, engine_name, teacher, cache, stopping, worker_pids
):
    global _DIRECT_CONFIG, _DIRECT_ENGINE, _DIRECT_ENGINE_PATH, _DIRECT_NAME
    global _DIRECT_TEACHER, _DIRECT_CACHE, _DIRECT_STOPPING
    _DIRECT_CONFIG = config
    _DIRECT_ENGINE_PATH = engine_path
    _DIRECT_ENGINE = None
    _DIRECT_CACHE = cache
    _DIRECT_STOPPING = stopping
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _interrupt_direct_worker)
    Finalize(None, _close_direct_engine, exitpriority=10)
    _DIRECT_NAME = engine_name
    _DIRECT_TEACHER = teacher
    with worker_pids.get_lock():
        worker_pids[worker_pids[:].index(0)] = os.getpid()


def _synthesize_groups(
    groups, config, engine_path, engine_name, teacher, provider, workers, cache, run
):
    """Refill a bounded set of root searches across shard boundaries.

    Yields ``(index, group, records, failures)`` per group, where ``failures``
    maps positions of uncollected roots to their last collection error. A
    group with failures is not publishable; its collected roots are
    checkpointed, so retrying it recollects only the missing roots.
    """
    if not groups:
        return
    if provider is not None or workers == 1:
        engine = None
        try:
            for index, group in groups:
                records = [None] * len(group)
                failures = {}
                for position, root in enumerate(group):
                    cached, capped, dropped = _cached_or_capped(cache, index, root)
                    if cached is not None:
                        records[position] = cached
                        continue
                    if capped is not None:
                        failures[position] = capped
                        continue
                    if dropped:
                        continue
                    if provider is None and engine is None:
                        engine = _start_engine(engine_path, config)
                    try:
                        record = _collect_root(
                            root, config, engine, engine_name, teacher, provider
                        )
                    except EngineTimeoutError:
                        _note_dropped(cache, index, root)
                        if engine is not None:
                            engine.close(force=True)
                            engine = None
                        continue
                    except CollectionError as error:
                        failures[position] = _note_failure(
                            cache, run, index, root, error
                        )
                        if engine is not None:
                            engine.close(force=True)
                            engine = None
                        continue
                    cache.store(index, root, record)
                    records[position] = record
                yield index, group, records, failures
        finally:
            if engine is not None:
                engine.close(force=True)
        return
    workers = min(workers, sum(len(group) for _, group in groups))
    pending = {}
    active = {}
    completed = deque()

    def finished(index):
        state = active[index]
        state["remaining"] -= 1
        if state["remaining"] == 0:
            completed.append(index)

    def missing_roots():
        for index, group in groups:
            active[index] = {"group": group, "remaining": len(group), "failures": {}}
            for position, root in enumerate(group):
                cached, capped, dropped = _cached_or_capped(cache, index, root)
                if cached is not None or capped is not None or dropped:
                    if capped is not None:
                        active[index]["failures"][position] = capped
                    del cached
                    finished(index)
                else:
                    yield index, position, root

    missing = missing_roots()
    context = get_context("spawn")
    stopping = context.Event()
    worker_pids = context.Array("i", workers)
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_direct_worker,
        initargs=(
            config,
            engine_path,
            engine_name,
            teacher,
            cache,
            stopping,
            worker_pids,
        ),
    )
    try:
        while True:
            while len(pending) < 2 * workers:
                item = next(missing, None)
                if item is None:
                    break
                index, position, root = item
                pending[pool.submit(_direct_worker, index, root)] = (index, position)
            while completed:
                index = completed.popleft()
                state = active.pop(index)
                group, failures = state["group"], state["failures"]
                records = []
                if not failures:
                    for root in group:
                        record = cache.load(index, root)
                        if record is None and cache.dropped(index, root) is None:
                            raise CollectionError(
                                f"Missing checkpoint for root {root['root_id']}"
                            )
                        records.append(record)
                yield index, group, records, failures
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index, position = pending.pop(future)
                state = active[index]
                group, failures = state["group"], state["failures"]
                root = group[position]
                try:
                    future.result()
                except EngineStartupError:
                    raise
                except EngineTimeoutError:
                    _note_dropped(cache, index, root)
                except CollectionError as error:
                    failures[position] = _note_failure(cache, run, index, root, error)
                finished(index)
    except BaseException:
        stopping.set()
        # Python 3.13 has no public executor method to interrupt workers.
        # Interrupt searches before joining, including engine startup waits.
        with worker_pids.get_lock():
            pids = worker_pids[:]
        for pid in pids:
            if pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _syzygy_tables(path):
    """A content fingerprint of the configured tablebase directory.

    Two runs probe identically only when the same tables are present, so the
    table manifest (names and sizes, not the host-specific path) joins the
    search identity; an unusable directory fails the run before any search.
    """
    tables = Path(path)
    files = (
        sorted(
            (entry.name, entry.stat().st_size)
            for pattern in ("*.rtbw", "*.rtbz")
            for entry in tables.glob(pattern)
        )
        if tables.is_dir()
        else []
    )
    if not any(name.endswith(".rtbw") for name, _ in files):
        raise ValueError(f"Syzygy tablebases are unusable: {path}")
    return hashlib.sha256(json.dumps(files).encode("utf-8")).hexdigest()


def _search_identity(config):
    identity = config.search_dict()
    identity["syzygy_tables"] = (
        _syzygy_tables(config.search.syzygy_path) if config.search.syzygy_path else None
    )
    return identity


def _identity_for(config, roots, engine_name="stockfish", root_source=None):
    return {
        "teacher": {
            "schema_version": (
                SIDECAR_SCHEMA_WIDE
                if any(root["teacher_width"] > 1 for root in roots)
                else SIDECAR_SCHEMA
            ),
            "value_mate": 32000,
            "max_ply": 246,
        },
        "engine": {"name": engine_name},
        "search": _search_identity(config),
        "board": config.board_dict(),
        "presentation": config.presentation_dict(),
        "synthesis": config.synthesis_dict(),
        "root_selection": {
            "source": str(root_source) if root_source is not None else "in-memory",
            "algorithm": "stratified",
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
    engine="stockfish",
    root_source=None,
):
    """Produce shards directly from roots, with optional fixture sidecars.

    ``sidecar_for_root`` is a callable/mapping used by tests and offline
    converters. When it is omitted, each worker resets its instrumented
    engine for each root; sidecars exist only in temporary directories.

    Every collected root is checkpointed under ``<output>/.roots`` until its
    shard is registered, so an interrupted run recollects only missing roots.
    A root whose engine exceeds the per-root timeout is dropped without
    retry and recorded under ``dropped`` in the release. Any other
    ``CollectionError`` is isolated from its shard mates, retried in a
    second pass, and retried across runs up to ``MAX_ROOT_ATTEMPTS``; if any
    roots remain failed after everything else is published, the run raises
    one summary error and keeps the checkpoints. The cache directory is
    removed once the release is complete.
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
        engine_path = _engine_path(engine)
        engine_name = "Stockfish 17.1"
        teacher = (32000, 246)
    identity = identity or _identity_for(
        config, roots, engine_name=engine_name, root_source=root_source
    )
    release_metadata = _release_metadata(identity)
    shard_metadata = {
        key: value
        for key, value in release_metadata.items()
        if key not in RELEASE_ONLY_METADATA
    }
    presentations = {root["root_id"]: (root["presentation"],) for root in roots}
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
    cache = _RootCache(output_dir, identity)
    ready, pending = [], []
    dropped_by_shard = {}
    for entry in (previous or {}).get("dropped", []):
        dropped_by_shard.setdefault(entry["shard"], set()).add(entry["root_id"])
    for index, group in groups:
        if index < start_shard or (end_shard is not None and index >= end_shard):
            continue
        if index in shards_by_index:
            dropped = dropped_by_shard.get(index, set())
            _check_shard_layout(
                output_dir,
                shards_by_index[index],
                [root for root in group if root["root_id"] not in dropped],
            )
            cache.clear_shard(index)
            continue
        final = output_dir / f"shard-{index:06d}"
        if final.exists():
            dropped = [
                root["root_id"]
                for root in group
                if cache.dropped(index, root) is not None
            ]
            shard = _recover_shard(
                output_dir,
                index,
                [root for root in group if root["root_id"] not in set(dropped)],
                shard_metadata,
                presentations,
            )
            ready.append((index, group, shard, dropped))
        else:
            pending.append((index, group))

    run = {"failures": 0}
    failed = []

    def publishable():
        for index, group, shard, dropped in ready:
            yield index, group, shard, dropped
        deferred = []
        for index, group, records, failures in _synthesize_groups(
            pending,
            config,
            engine_path,
            engine_name,
            teacher,
            sidecar_for_root,
            workers,
            cache,
            run,
        ):
            if failures:
                deferred.append((index, group))
                continue
            dropped = [
                group[position]["root_id"]
                for position, record in enumerate(records)
                if record is None
            ]
            yield (
                index,
                group,
                _publish_shard(
                    output_dir, index, records, shard_metadata, presentations
                ),
                dropped,
            )
        # A deferred group's collected roots are checkpoint hits here; only
        # the failed roots are recollected in this second pass.
        for index, group, records, failures in _synthesize_groups(
            deferred,
            config,
            engine_path,
            engine_name,
            teacher,
            sidecar_for_root,
            workers,
            cache,
            run,
        ):
            if failures:
                failed.extend(
                    (index, group[position]["root_id"], error)
                    for position, error in failures.items()
                )
                continue
            dropped = [
                group[position]["root_id"]
                for position, record in enumerate(records)
                if record is None
            ]
            yield (
                index,
                group,
                _publish_shard(
                    output_dir, index, records, shard_metadata, presentations
                ),
                dropped,
            )

    for index, group, shard, dropped_ids in publishable():
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
            if index in shards_by_index:
                # Shard ranges are disjoint by contract; a second producer
                # targeting the same shard must fail loudly, not mix data.
                raise FileExistsError(f"Shard is already registered: {index:06d}")
            shards_by_index[index] = shard
            dropped = (current or {}).get("dropped", []) + [
                {"root_id": root_id, "shard": index} for root_id in dropped_ids
            ]
            write_manifest(
                release_path,
                {
                    **release_metadata,
                    "metadata": identity,
                    "dropped": dropped,
                    "shards": [shards_by_index[key] for key in sorted(shards_by_index)],
                },
            )
        cache.clear_shard(index)
    with _release_lock(output_dir):
        current = load_manifest(release_path)
        complete = sum(shard["rows"] for shard in current.get("shards", []))
        dropped = current.get("dropped", [])
        profile = RootProfile(presentations)
        for shard in current.get("shards", []):
            _profile_shard(output_dir, shard, profile)
        statistics = profile.result() if profile.roots else None
        if statistics is not None:
            statistics["complete"] = complete + len(dropped) == len(roots)
            write_manifest(output_dir / "statistics.json", statistics)
    if failed:
        summary = "; ".join(
            f"shard {index:06d} root {root_id} ({error})"
            for index, root_id, error in failed[:10]
        )
        raise CollectionError(
            f"{len(failed)} roots failed permanently; their shards stay "
            "unpublished and every collected root stays checkpointed for the "
            f"next run: {summary}"
        )
    if complete + len(dropped) == len(roots):
        # A partial-range run keeps the cache for shards it never touched.
        shutil.rmtree(cache.directory, ignore_errors=True)
    return {
        "shards": current.get("shards", []),
        "roots": complete,
        "dropped": len(dropped),
        "pending": len(roots) - complete - len(dropped),
        "release": str(release_path),
        "statistics": statistics,
    }


def run_pipeline(config, roots, output, engine="stockfish"):
    """Curate traces from a prepared pool without modifying its records."""
    from pondera.data.roots.pool import read_roots

    roots = Path(roots)
    if roots.is_dir():
        roots /= "roots.jsonl"
    return produce_shards(
        read_roots(roots),
        output,
        config,
        start_shard=config.release.start_shard,
        end_shard=config.release.end_shard,
        engine=engine,
        root_source=roots,
    )
