"""Translate instrumented Stockfish records into grammar-v2 sequence traces.

The sidecar is the only source of teacher statements. Synthesis never asks the
engine for a value: every written evaluation comes from an exact call, every
refutation from a null-window fail-low, and every frontier value from the
recorded quiescence return. Failed aspiration attempts, bounds at internal
nodes, and candidates whose final call returned a bound stay in the record and
are not written; the teacher trace therefore contains no passes.

A complete tool trace is synthesized once per root, including both BOARD
sources. The four presentations are prefixes of that trace plus one shared
label suffix, so all presentations of a root predict the same final move,
evaluation, and variation.
"""

import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

from pondera.chess.codec import mate_token, value_bin
from pondera.chess.symmetry import reflect_uci
from pondera.chess.tracker import PathTracker
from pondera.chess.vocabulary import (
    DEPTH,
    FINAL,
    MATE_LOSS,
    MATE_WIN,
    MOVE,
    PAD,
    UCI_MOVE_TO_IDX,
    VALUE,
)
from pondera.data.trace import CLAIM_DRAW_UCI, DRAW_CLAIM_BOUND, TraceWriter

SIDECAR_SCHEMA = 3
SIDECAR_SCHEMA_WIDE = 4  # MultiPV teacher: per-line attempts (docs/teacher.md)
PRESENTATIONS = ("zero", "full", "random", "last_root")
EVENT_TYPES = frozenset(
    {
        "session",
        "position",
        "search_start",
        "round_start",
        "attempt",
        "candidate",
        "frontier",
        "round_end",
        "search_end",
    }
)
CALL_ROLES = frozenset({"first", "probe", "rescore", "promote"})
BOUNDS = frozenset({"exact", "upper", "lower"})


class SynthesisError(ValueError):
    """A record that cannot be translated without inventing information."""


def _reject_float(value):
    raise SynthesisError(f"sidecar contains a floating-point number: {value}")


def _number(value, name, minimum=None, maximum=None):
    if not isinstance(value, int) or isinstance(value, bool):
        raise SynthesisError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise SynthesisError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise SynthesisError(f"{name} must be at most {maximum}")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise SynthesisError(f"{name} must be a nonempty string")
    return value


def _window(value, name):
    if not isinstance(value, dict) or set(value) != {"alpha", "beta"}:
        raise SynthesisError(f"{name} must be an alpha/beta window")
    alpha = _number(value["alpha"], f"{name}.alpha")
    beta = _number(value["beta"], f"{name}.beta")
    if alpha > beta:
        raise SynthesisError(f"{name} has alpha above beta")
    return alpha, beta


def _is_mate(raw, value_mate, max_ply):
    return abs(raw) >= value_mate - max_ply


def _check_cp_wdl(event, value_mate, max_ply, name):
    raw = _number(event["raw"], f"{name}.raw")
    mate = _is_mate(raw, value_mate, max_ply)
    if mate:
        if "cp" in event or "wdl" in event:
            raise SynthesisError(f"{name} has cp/wdl on a mate score")
        return raw, None, None
    if "cp" not in event or "wdl" not in event:
        raise SynthesisError(f"{name} is missing cp/wdl on an ordinary score")
    cp = _number(event["cp"], f"{name}.cp")
    wdl = event["wdl"]
    if (
        not isinstance(wdl, list)
        or len(wdl) != 3
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= 1000
            for value in wdl
        )
        or sum(wdl) != 1000
    ):
        raise SynthesisError(
            f"{name}.wdl must be three per-mille integers summing to 1000"
        )
    return raw, cp, tuple(wdl)


def _bound_of(raw, alpha, beta):
    if raw <= alpha:
        return "upper"
    if raw >= beta:
        return "lower"
    return "exact"


def _path(value, name):
    if not isinstance(value, list) or not value:
        raise SynthesisError(f"{name} must be a nonempty move list")
    moves = []
    for move in value:
        if not isinstance(move, str):
            raise SynthesisError(f"{name} contains a non-string move")
        moves.append(move)
    return tuple(moves)


def _legal_line(root_fen, repetition, moves, stop_at_terminal=True):
    tracker = PathTracker(root_fen, repetition)
    for index, move in enumerate(moves):
        remaining = index + 1 < len(moves)
        try:
            tracker.push(move)
        except ValueError as error:
            raise SynthesisError(
                f"illegal move in a principal variation: {move}"
            ) from error
        if stop_at_terminal and remaining and tracker.search_terminal() is not None:
            raise SynthesisError("principal variation continues past a rule terminal")
    return tracker


@dataclass(frozen=True)
class Sidecar:
    """A validated schema-3 or schema-4 record, indexed by event type."""

    events: tuple
    session: dict
    position: dict
    search_start: dict
    round_starts: tuple
    attempts: tuple
    candidates: tuple
    frontiers: tuple
    round_ends: tuple
    search_end: dict

    @property
    def fen(self):
        return self.position["fen"]

    @property
    def width(self):
        """The effective teacher width: min(configured width, legal root moves)."""
        return self.search_start["multi_pv"]

    @property
    def configured_width(self):
        """The MultiPV value the engine was configured with for this root."""
        return self.session["options"]["multipv"]

    @property
    def value_mate(self):
        return self.session["value_mate"]

    @property
    def max_ply(self):
        return self.session["max_ply"]

    @property
    def depth(self):
        return self.search_end["depth"]

    @property
    def engine(self):
        return self.session["engine"]


def parse_sidecar(lines, teacher_width=1, source="<sidecar>"):
    """Parse and independently validate one sidecar JSONL stream.

    ``teacher_width`` is the configured MultiPV value. The engine caps each
    search at the number of legal root moves and records the capped value as
    the effective width; ``Sidecar.width`` exposes it.
    """
    if teacher_width < 1:
        raise SynthesisError("teacher width must be at least one")
    expected_schema = SIDECAR_SCHEMA if teacher_width == 1 else SIDECAR_SCHEMA_WIDE
    events = []
    for index, line in enumerate(lines, 1):
        if not line.strip():
            raise SynthesisError(f"{source}:{index}: blank line")
        try:
            event = json.loads(line, parse_float=_reject_float)
        except json.JSONDecodeError as error:
            raise SynthesisError(f"{source}:{index}: invalid JSON") from error
        if not isinstance(event, dict):
            raise SynthesisError(f"{source}:{index}: event is not an object")
        if event.get("schema") != expected_schema:
            raise SynthesisError(f"{source}:{index}: unsupported schema")
        if event.get("seq") != index:
            raise SynthesisError(
                f"{source}:{index}: seq does not equal the line number"
            )
        if event.get("type") not in EVENT_TYPES:
            raise SynthesisError(f"{source}:{index}: unknown event type")
        events.append(event)
    if not events:
        raise SynthesisError(f"{source}: empty sidecar")
    if events[0]["type"] != "session" or events[-1]["type"] != "search_end":
        raise SynthesisError(f"{source}: session must be first and search_end last")
    for kind in ("session", "position", "search_start", "search_end"):
        count = sum(1 for event in events if event["type"] == kind)
        if count != 1:
            raise SynthesisError(f"{source}: expected one {kind}, found {count}")
    session = events[0]
    position = next(event for event in events if event["type"] == "position")
    search_start = next(event for event in events if event["type"] == "search_start")
    search_end = events[-1]

    _text(session.get("engine"), "session.engine")
    value_mate = _number(session.get("value_mate"), "session.value_mate", 1)
    max_ply = _number(session.get("max_ply"), "session.max_ply", 1)
    options = session.get("options")
    if not isinstance(options, dict) or set(options) != {
        "threads",
        "hash_mb",
        "multipv",
    }:
        raise SynthesisError("session.options is incomplete")
    _number(options["threads"], "session.options.threads", 1)
    _number(options["hash_mb"], "session.options.hash_mb", 1)
    _number(options["multipv"], "session.options.multipv", 1)
    if options["threads"] != 1:
        raise SynthesisError("a recorded search must use one thread")
    if options["multipv"] != teacher_width:
        raise SynthesisError("session.options.multipv differs from the teacher width")

    fen = _text(position.get("fen"), "position.fen")
    try:
        root_board = chess.Board(fen)
    except ValueError as error:
        raise SynthesisError("position.fen is not a valid FEN") from error
    if root_board.is_checkmate() or root_board.is_stalemate():
        raise SynthesisError("a terminal root is not a valid trace root")
    # The engine caps MultiPV at the number of legal root moves and records the
    # capped value as the effective width of the search.
    effective_width = min(teacher_width, root_board.legal_moves.count())
    depth = _number(search_start.get("depth"), "search_start.depth", 1)
    _number(search_start.get("multi_pv"), "search_start.multi_pv", 1)
    if search_start["multi_pv"] != effective_width:
        raise SynthesisError(
            "search_start.multi_pv differs from the capped teacher width"
        )
    # The schema and the presence of line numbers follow the configured width;
    # the line and rank bounds follow the effective width.
    configured_wide = teacher_width > 1

    round_starts = [event for event in events if event["type"] == "round_start"]
    attempts = [event for event in events if event["type"] == "attempt"]
    candidates = [event for event in events if event["type"] == "candidate"]
    frontiers = [event for event in events if event["type"] == "frontier"]
    round_ends = [event for event in events if event["type"] == "round_end"]
    if not round_ends:
        raise SynthesisError("the search has no completed round")
    if len(round_starts) != len(round_ends):
        raise SynthesisError("round_start and round_end counts differ")
    rounds = [event["round"] for event in round_ends]
    if rounds != list(range(1, len(round_ends) + 1)):
        raise SynthesisError("rounds are not contiguous from one")
    start_rounds = [event["round"] for event in round_starts]
    if start_rounds != rounds:
        raise SynthesisError("round_start rounds do not match round_end rounds")
    for start, end in zip(round_starts, round_ends):
        _number(start.get("round"), "round_start.round", 1)
        _number(start.get("depth"), "round_start.depth", 1)
        _window(start.get("window"), "round_start.window")
        if start["round"] != end["round"] or start["depth"] != end["depth"]:
            raise SynthesisError("round_start and round_end disagree on depth")
        if start["depth"] != start["round"]:
            raise SynthesisError("iteration depth does not equal the round number")
        if not isinstance(start.get("root_order"), list) or not start["root_order"]:
            raise SynthesisError("round_start.root_order is empty")
        for move in start["root_order"]:
            if not isinstance(move, str):
                raise SynthesisError(
                    "round_start.root_order contains a non-string move"
                )
    if search_end.get("depth") != round_ends[-1]["depth"]:
        raise SynthesisError("search_end.depth is not the deepest completed round")
    if round_ends[-1]["depth"] > depth:
        raise SynthesisError("a round is deeper than the requested depth")

    attempts_by_round = {round_number: [] for round_number in rounds}
    accepted_by_round = {}
    for event in attempts:
        round_number = _number(event.get("round"), "attempt.round", 1)
        if round_number not in attempts_by_round:
            raise SynthesisError("attempt references an unknown round")
        _number(event.get("attempt"), "attempt.attempt", 1)
        line = event.get("line")
        if configured_wide:
            if line is None:
                raise SynthesisError("a wide-teacher attempt has no line")
            _number(line, "attempt.line", 0, effective_width - 1)
        elif line is not None:
            raise SynthesisError("a width-one attempt carries a line")
        alpha, beta = _window(event.get("window"), "attempt.window")
        raw, cp, wdl = _check_cp_wdl(event, value_mate, max_ply, "attempt")
        bound = event.get("bound")
        if bound not in BOUNDS or bound != _bound_of(raw, alpha, beta):
            raise SynthesisError("attempt.bound disagrees with its window and value")
        attempts_by_round[round_number].append(event)
    for round_event, end_event in zip(round_starts, round_ends):
        round_number = end_event["round"]
        round_attempts = attempts_by_round[round_number]
        if not round_attempts:
            raise SynthesisError(f"round {round_number} has no attempt")
        numbers = [event["attempt"] for event in round_attempts]
        if numbers != list(range(1, len(numbers) + 1)):
            raise SynthesisError(
                f"round {round_number} attempts are not contiguous from one"
            )
        _number(end_event.get("attempt"), "round_end.attempt", 1)
        if configured_wide:
            if end_event["attempt"] != len(round_attempts):
                raise SynthesisError(
                    f"round {round_number} does not count its attempts"
                )
            by_line = {}
            for event in round_attempts:
                by_line.setdefault(event["line"], []).append(event)
            if sorted(by_line) != list(range(len(by_line))):
                raise SynthesisError(
                    f"round {round_number} lines are not contiguous from zero"
                )
            if len(by_line) != effective_width:
                raise SynthesisError(
                    f"round {round_number} does not run its effective width"
                )
            for line, line_attempts in by_line.items():
                exact = [event for event in line_attempts if event["bound"] == "exact"]
                if len(exact) != 1 or exact[0] is not line_attempts[-1]:
                    raise SynthesisError(
                        f"round {round_number} line {line} does not accept "
                        "exactly its last attempt"
                    )
        else:
            exact = [event for event in round_attempts if event["bound"] == "exact"]
            if len(exact) != 1:
                raise SynthesisError(
                    f"round {round_number} has {len(exact)} exact attempts"
                )
            if end_event.get("attempt") != exact[0]["attempt"]:
                raise SynthesisError(
                    f"round {round_number} does not accept its exact attempt"
                )
        if configured_wide:
            accepted_by_round[round_number] = {
                line: line_attempts[-1] for line, line_attempts in by_line.items()
            }
        else:
            accepted_by_round[round_number] = {0: exact[0]}
        rank = end_event.get("rank")
        if not isinstance(rank, list) or len(rank) != effective_width:
            raise SynthesisError(f"round {round_number} rank is invalid")
        previous_raw = None
        for entry in rank:
            if not isinstance(entry, dict):
                raise SynthesisError("round_end.rank entry is not an object")
            _text(entry.get("move"), "round_end.rank.move")
            raw, cp, wdl = _check_cp_wdl(entry, value_mate, max_ply, "round_end.rank")
            if previous_raw is not None and raw > previous_raw:
                raise SynthesisError("round_end.rank is not sorted by raw")
            previous_raw = raw
            pv = entry.get("pv")
            if not isinstance(pv, list) or not pv or pv[0] != entry["move"]:
                raise SynthesisError("round_end.rank.pv does not start with its move")
            _legal_line(fen, 1, [str(move) for move in pv])
        top = rank[0]
        top_raw, top_cp, top_wdl = _check_cp_wdl(
            top, value_mate, max_ply, "round_end.rank[0]"
        )
        establishing = max(
            accepted_by_round[round_number].values(), key=lambda event: event["raw"]
        )
        if top_raw != establishing["raw"]:
            raise SynthesisError("rank[0].raw differs from the accepted attempts")
        if "cp" in establishing and top_cp != establishing["cp"]:
            raise SynthesisError("rank[0].cp differs from the accepted attempts")
        if "wdl" in establishing and list(top_wdl) != list(establishing["wdl"]):
            raise SynthesisError("rank[0].wdl differs from the accepted attempts")

    # Every candidate and frontier belongs to an attempt of its round.
    for event in events:
        kind = event["type"]
        if kind not in {"candidate", "frontier"}:
            continue
        round_number = _number(event.get("round"), f"{kind}.round", 1)
        if round_number not in attempts_by_round:
            raise SynthesisError(f"{kind} references an unknown round")
        attempt_number = _number(event.get("attempt"), f"{kind}.attempt", 1)
        if attempt_number not in {
            attempt["attempt"] for attempt in attempts_by_round[round_number]
        }:
            raise SynthesisError(f"{kind} references an unknown attempt")
        path = _path(event.get("path"), f"{kind}.path")
        if event.get("ply") != len(path):
            raise SynthesisError(f"{kind}.ply does not equal its path length")
        tracker = PathTracker(fen, 1)
        try:
            for move in path:
                tracker.push(move)
        except ValueError as error:
            raise SynthesisError(f"{kind} contains an illegal move") from error
        if kind == "candidate":
            if event.get("move") != path[-1]:
                raise SynthesisError("candidate.move is not the last path element")
            _number(event.get("order"), "candidate.order", 1)
            call = event.get("call")
            if call not in CALL_ROLES:
                raise SynthesisError("candidate.call is unknown")
            _number(event.get("depth"), "candidate.depth")
            alpha, beta = _window(event.get("window"), "candidate.window")
            if call in {"probe", "rescore"} and beta - alpha != 1:
                raise SynthesisError("a probe or rescore must use a null window")
            raw, cp, wdl = _check_cp_wdl(event, value_mate, max_ply, "candidate")
            bound = event.get("bound")
            if bound not in BOUNDS or bound != _bound_of(raw, alpha, beta):
                raise SynthesisError(
                    "candidate.bound disagrees with its window and value"
                )
            _number(event.get("nodes_delta"), "candidate.nodes_delta", 0)
            _number(event.get("seldepth"), "candidate.seldepth", 0)
            if event["ply"] == 1 and call in {"first", "promote"}:
                if "pv" not in event:
                    raise SynthesisError("a root full-window candidate has no pv")
                if not isinstance(event["pv"], list) or not event["pv"]:
                    raise SynthesisError("candidate.pv is empty")
                if event["pv"][0] != event["move"]:
                    raise SynthesisError("candidate.pv does not start with its move")
                _legal_line(fen, 1, event["pv"])
            elif "pv" in event:
                raise SynthesisError("a non-root or narrow candidate carries a pv")
        else:
            raw, cp, wdl = _check_cp_wdl(event, value_mate, max_ply, "frontier")
            bound = event.get("bound")
            if bound not in BOUNDS:
                raise SynthesisError("frontier.bound is unknown")
            _number(event.get("nodes_delta"), "frontier.nodes_delta", 0)
            _number(event.get("seldepth"), "frontier.seldepth", 0)

    # A frontier duplicates the value of the first/promote call that opened it.
    frontier_keys = {}
    for event in frontiers:
        key = (
            event["round"],
            event["attempt"],
            tuple(event["path"]),
        )
        if key in frontier_keys:
            raise SynthesisError("a path has more than one frontier event")
        frontier_keys[key] = event
    opening_calls = {}
    for event in candidates:
        if event["call"] in {"first", "promote"}:
            key = (event["round"], event["attempt"], tuple(event["path"]))
            opening_calls.setdefault(key, []).append(event)
    for key, frontier in frontier_keys.items():
        matches = opening_calls.get(key, [])
        if not matches:
            raise SynthesisError("a frontier has no first/promote candidate")
        if not any(
            call["raw"] == frontier["raw"]
            and call.get("cp") == frontier.get("cp")
            and call.get("wdl") == frontier.get("wdl")
            and call["bound"] == frontier["bound"]
            for call in matches
        ):
            raise SynthesisError("a frontier disagrees with its opening call")

    for end_event in round_ends:
        round_number = end_event["round"]
        for event in events:
            if event.get("round") == round_number and event is not end_event:
                if event.get("seq", 0) > end_event["seq"] and event["type"] in {
                    "candidate",
                    "frontier",
                    "attempt",
                }:
                    raise SynthesisError(
                        f"round {round_number} has events after its end"
                    )

    # A root null-window fail-high must be re-searched in the same attempt.
    for end_event in round_ends:
        round_number = end_event["round"]
        accepted = {
            event["attempt"] for event in accepted_by_round[round_number].values()
        }
        groups = {}
        for event in candidates:
            if event["round"] == round_number and event["attempt"] in accepted:
                groups.setdefault((tuple(event["path"]), event["move"]), []).append(
                    event
                )
        for (path, _), calls in groups.items():
            if len(path) != 1:
                continue
            last = calls[-1]
            if (
                last["call"] in {"probe", "rescore"}
                and last["bound"] == "upper"
                and not any(
                    call["call"] in {"first", "promote"} and call["bound"] == "exact"
                    for call in calls
                )
            ):
                raise SynthesisError(
                    "a root null-window fail-high has no full-window follow-up"
                )

    # Every rank entry has an exact establishing call in an accepted pass.
    for end_event in round_ends:
        round_number = end_event["round"]
        accepted = {
            event["attempt"] for event in accepted_by_round[round_number].values()
        }
        for top in end_event["rank"]:
            establishing = [
                event
                for event in candidates
                if event["round"] == round_number
                and event["attempt"] in accepted
                and event["ply"] == 1
                and event["move"] == top["move"]
                and event["call"] in {"first", "promote"}
                and event["bound"] == "exact"
            ]
            if not establishing:
                raise SynthesisError("a rank entry has no exact establishing call")
            call = establishing[-1]
            if call["raw"] != -top["raw"]:
                raise SynthesisError("an establishing call does not negate its rank")
            if call.get("pv") != top.get("pv"):
                raise SynthesisError("an establishing call pv differs from its rank")
            # No wdl symmetry is asserted here: candidate wdl is child-relative and
            # rank wdl root-relative, and the engine's win-rate model scales with
            # the material of the position it describes, so a capture's child wdl
            # is not the exact mirror of the root wdl (docs/teacher.md).

    if search_end.get("bestmove") is None:
        raise SynthesisError("search_end has no bestmove")
    if search_end["bestmove"] != round_ends[-1]["rank"][0]["move"]:
        raise SynthesisError("search_end.bestmove differs from the deepest rank[0]")
    _number(search_end.get("nodes"), "search_end.nodes", 0)
    _number(search_end.get("seldepth"), "search_end.seldepth", 0)

    return Sidecar(
        tuple(events),
        session,
        position,
        search_start,
        tuple(round_starts),
        tuple(attempts),
        tuple(candidates),
        tuple(frontiers),
        tuple(round_ends),
        search_end,
    )


def _sidecar_configured_width(lines, source):
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise SynthesisError(f"{source}:{index}: invalid JSON") from error
        if not isinstance(event, dict) or event.get("type") != "session":
            raise SynthesisError(f"{source}: session must be first")
        options = event.get("options")
        width = options.get("multipv") if isinstance(options, dict) else None
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise SynthesisError(f"{source}: session has no valid multipv")
        return width
    raise SynthesisError(f"{source}: empty sidecar")


def load_sidecar(path, teacher_width=None):
    """Load a sidecar, inferring the configured width from its session line."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if teacher_width is None:
        teacher_width = _sidecar_configured_width(lines, str(path))
    return parse_sidecar(
        lines,
        teacher_width=teacher_width,
        source=str(path),
    )


def mate_distance_moves(raw, value_mate, max_ply):
    """Convert a root-player mate score from plies to UCI move distance.

    This is the ``UCIEngine::format_score`` rule: a win rounds the remaining
    plies up by one before halving; a loss truncates toward zero before the
    sign is applied.
    """
    if not _is_mate(raw, value_mate, max_ply):
        raise ValueError("Not a mate score")
    distance = value_mate - abs(raw)
    if raw > 0:
        return (distance + 1) // 2
    return -(distance // 2)


def _token_mate(token):
    if token >= MATE_LOSS:
        return -(token - MATE_LOSS)
    if token >= MATE_WIN:
        return token - MATE_WIN
    return None


def _thought_token(raw, wdl, ply, value_mate, max_ply):
    if wdl is not None:
        scalar = (wdl[0] - wdl[2]) / 1000
        if ply % 2:
            scalar = -scalar
        return VALUE + value_bin(scalar)
    root_raw = -raw if ply % 2 else raw
    moves = mate_distance_moves(root_raw, value_mate, max_ply)
    return mate_token(abs(moves), moves > 0)


def _final_parameters(raw, wdl, value_mate, max_ply):
    if wdl is not None:
        return {"value": (wdl[0] - wdl[2]) / 1000}
    moves = mate_distance_moves(raw, value_mate, max_ply)
    return {"mate": abs(moves), "win": moves > 0}


@dataclass(frozen=True)
class Boundary:
    tool_end: int
    no_tool_end: int
    best: int
    value: int
    basis: int


@dataclass(frozen=True)
class CandidateAudit:
    round: int
    move: str
    raw: int
    wdl: tuple | None
    mate: int | None
    value_token: int
    order: int
    draw: bool


@dataclass(frozen=True)
class RoundAudit:
    round: int
    depth: int
    move: str
    raw: int
    wdl: tuple | None
    mate: int | None
    value_token: int
    pv: tuple
    candidates: tuple
    refutations: int


@dataclass(frozen=True)
class PresentationRow:
    presentation: str
    tokens: tuple
    roles: tuple
    boundaries: tuple
    rounds: int
    thinking_tokens: int
    final_move: str
    final_value_bin: int
    mate_distance: int | None
    claim_draw: bool


@dataclass(frozen=True)
class SynthesizedRoot:
    root: dict
    rows: dict
    round_audits: tuple
    stats: dict


def _move_token(uci, mirrored):
    framed = reflect_uci(uci) if mirrored else uci
    return MOVE + UCI_MOVE_TO_IDX[framed]


class _Candidate:
    __slots__ = ("move", "parent", "order", "calls", "exact", "refuted", "first_seq")

    def __init__(self, move, parent, order, calls):
        self.move = move
        self.parent = parent
        self.order = order
        self.calls = calls
        self.first_seq = calls[0]["seq"]
        exact = [
            call
            for call in calls
            if call["call"] in {"first", "promote"} and call["bound"] == "exact"
        ]
        self.exact = exact[-1] if exact else None
        refuted = [
            call
            for call in calls
            if call["call"] in {"probe", "rescore"} and call["bound"] == "lower"
        ]
        self.refuted = refuted[-1] if refuted and self.exact is None else None


def _candidate_groups(sidecar, round_number, attempt):
    groups = {}
    for event in sidecar.candidates:
        if event["round"] != round_number or event["attempt"] != attempt:
            continue
        parent = tuple(event["path"][:-1])
        key = (parent, event["move"])
        groups.setdefault(key, []).append(event)
    result = {}
    for (parent, move), calls in groups.items():
        orders = {call["order"] for call in calls}
        if len(orders) != 1:
            raise SynthesisError("calls for one candidate disagree on order")
        result.setdefault(parent, []).append(
            _Candidate(move, parent, orders.pop(), calls)
        )
    for values in result.values():
        values.sort(key=lambda candidate: (candidate.order, candidate.first_seq))
    return result


def _frontier_for(sidecar, round_number, attempt, path):
    matches = [
        event
        for event in sidecar.frontiers
        if event["round"] == round_number
        and event["attempt"] == attempt
        and tuple(event["path"]) == path
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise SynthesisError("a frontier path has more than one frontier event")
    return matches[0]


class _RootState:
    """The champion visible at a truncation boundary."""

    def __init__(self, mirrored, teacher_width):
        self.mirrored = mirrored
        self.teacher_width = teacher_width
        self.round = 0
        self.current_round = {}
        self.round_sequence = {}
        self.round_count = 0
        self.previous = None
        self.current = None

    def begin_round(self, round_number):
        self.round = round_number
        self.current_round = {}
        self.round_sequence = {}
        self.round_count = 0
        self.current = self.previous

    def record_exact(self, move, raw, token):
        sequence = self.round_sequence.get(move)
        if sequence is None:
            self.round_count += 1
            sequence = self.round_count
            self.round_sequence[move] = sequence
        self.current_round[move] = (raw, token, sequence)
        # First to arrive defends the maximum: later equal values do not
        # replace the current champion, and only a strict gain does.
        best_move, best = max(
            self.current_round.items(),
            key=lambda item: (item[1][0], -item[1][2]),
        )
        self.current = (_move_token(best_move, self.mirrored), best[1], best[0])

    def finish_round(self, move, raw, token):
        # The round-end root evaluation refreshes the visible value even when
        # the champion move keeps its place on an equal raw score.
        best = (
            _move_token(move, self.mirrored)
            if self.current is None or raw > self.current[2]
            else self.current[0]
        )
        self.previous = (best, token, raw)
        self.current = self.previous

    def snapshot(self):
        if self.current is None:
            return -1, -1, self.round
        best, value, _ = self.current
        return best, value, self.round


class _Synthesis:
    def __init__(self, sidecar, root, config, seed):
        self.sidecar = sidecar
        self.root = root
        self.config = config
        self.seed = config.smoke.seed if seed is None else seed
        self.writer = TraceWriter(
            sidecar.fen,
            int(root["repetition"]),
            teacher_width=sidecar.width,
        )
        self.tracker = PathTracker(sidecar.fen, int(root["repetition"]))
        self.state = _RootState(self.writer.mirrored, sidecar.width)
        self.board_rng = random.Random(f"{self.seed}:{root['root_id']}:board")
        self.boundaries = []
        self.previous_len = 0
        self.no_tool_len = 0
        self.plies_since_board = 0
        self.board_counts = {"descent": 0, "frontier": 0}
        self.omitted = {"bound": 0, "early_refutation": 0, "root_fail_high": 0}
        self.exact_evals = 0
        self.refutations = 0
        self.terminal_evals = 0
        self.frontier_evals = 0
        self.round_audits = []

    # --- trace plumbing --- #

    def _record(self):
        length = len(self.writer.tokens)
        delta = length - self.previous_len
        operation = self.writer.steps[-1].operation
        if operation != "board":
            self.no_tool_len += delta
        best, value, basis = self.state.snapshot()
        self.boundaries.append(Boundary(length, self.no_tool_len, best, value, basis))
        self.previous_len = length

    def _write_move(self, move, spine=True):
        self.writer.move(move)
        self.tracker.push(move)
        self._record()
        if not spine:
            return
        self.plies_since_board += 1
        if self.board_rng.random() < self.config.board.board_descent_hazard:
            self.writer.board()
            self.board_counts["descent"] += 1
            self.plies_since_board = 0
            self._record()

    def _write_eval(self, token):
        self.writer.apply(token)
        self._record()

    # --- candidate narration --- #

    def _record_root_candidate(self, round_number, candidate, token, draw=False):
        if candidate.parent != ():
            return
        raw = -candidate.exact["raw"]
        self.state.record_exact(candidate.move, raw, token)
        wdl = None if draw else candidate.exact.get("wdl")
        self._recorded_root_candidates.append(
            CandidateAudit(
                round_number,
                candidate.move,
                raw,
                None if wdl is None else (wdl[2], wdl[1], wdl[0]),
                None if draw else _token_mate(token),
                token,
                candidate.order,
                draw,
            )
        )

    def _write_exact_child(self, round_number, attempt, candidate, groups):
        depth = candidate.exact["depth"]
        child_path = candidate.parent + (candidate.move,)
        self._write_move(candidate.move)
        terminal = self.tracker.search_terminal()
        if terminal is not None:
            raw = candidate.exact["raw"]
            if terminal == "mate":
                if (
                    not _is_mate(raw, self.sidecar.value_mate, self.sidecar.max_ply)
                    or raw >= 0
                ):
                    raise SynthesisError(
                        "a checkmate child does not carry a mated engine score"
                    )
                token = mate_token(0, True)
            else:
                # A draw is a rule verdict, not an engine score. Deeper nodes
                # return VALUE_DRAW with the +-1 tie-break, but qsearch does
                # not detect stalemate, so a depth-zero stalemate leaf carries
                # its static evaluation instead.
                if depth > 0 and abs(raw) > 1:
                    raise SynthesisError(
                        "a non-leaf rule draw child has a non-draw score"
                    )
                token = VALUE + value_bin(0.0)
            self.terminal_evals += 1
            self._record_root_candidate(
                round_number, candidate, token, draw=terminal != "mate"
            )
            self._write_eval(token)
            return
        if depth <= 0:
            frontier = _frontier_for(self.sidecar, round_number, attempt, child_path)
            if frontier is None:
                raise SynthesisError("a depth-zero exact child has no frontier event")
            if frontier["raw"] != candidate.exact["raw"]:
                raise SynthesisError("frontier value differs from its candidate")
            if frontier.get("cp") != candidate.exact.get("cp"):
                raise SynthesisError("frontier cp differs from its candidate")
            if frontier.get("wdl") != candidate.exact.get("wdl"):
                raise SynthesisError("frontier wdl differs from its candidate")
            if self.board_rng.random() < self.config.board.board_frontier_p:
                self.writer.board()
                self.board_counts["frontier"] += 1
                self._record()
            token = _thought_token(
                frontier["raw"],
                frontier.get("wdl"),
                candidate.exact["ply"],
                self.sidecar.value_mate,
                self.sidecar.max_ply,
            )
            self.frontier_evals += 1
            self._record_root_candidate(round_number, candidate, token)
            self._write_eval(token)
            return
        children = groups.get(child_path, [])
        if children:
            self._write_children(round_number, attempt, child_path, children, groups)
        token = _thought_token(
            candidate.exact["raw"],
            candidate.exact.get("wdl"),
            candidate.exact["ply"],
            self.sidecar.value_mate,
            self.sidecar.max_ply,
        )
        self.exact_evals += 1
        self._record_root_candidate(round_number, candidate, token)
        self._write_eval(token)

    def _write_children(self, round_number, attempt, parent, candidates, groups):
        needed = self.sidecar.width if parent == () else 1
        exact_written = 0
        for candidate in candidates:
            if candidate.exact is not None:
                self._write_exact_child(round_number, attempt, candidate, groups)
                exact_written += 1
                if parent == ():
                    self.writer.reset()
                    self.tracker.reset()
                else:
                    self.writer.back()
                    self.tracker.back()
                self._record()
                continue
            if candidate.refuted is not None:
                if exact_written < needed:
                    self.omitted["early_refutation"] += 1
                    continue
                self._write_move(candidate.move, spine=False)
                self.writer.refuted()
                self.refutations += 1
                self.current_refutations += 1
                self._record()
                if parent == ():
                    self.writer.reset()
                    self.tracker.reset()
                else:
                    self.writer.back()
                    self.tracker.back()
                self._record()
                continue
            self.omitted["bound"] += 1

    def _write_wide_round(self, round_number):
        attempts = [
            event for event in self.sidecar.attempts if event["round"] == round_number
        ]
        accepted = {}
        for event in attempts:
            accepted[event["line"]] = event["attempt"]
        # A root candidate is narrated once, from the accepted pass of the
        # line that produced its last exact call of the round.
        last_exact = {}
        for attempt in accepted.values():
            for candidate in _candidate_groups(self.sidecar, round_number, attempt).get(
                (), []
            ):
                if candidate.exact is not None:
                    last_exact[candidate.move] = attempt
        for line in sorted(accepted):
            attempt = accepted[line]
            groups = _candidate_groups(self.sidecar, round_number, attempt)
            for candidate in groups.get((), []):
                if candidate.exact is None or last_exact[candidate.move] != attempt:
                    continue
                self._write_exact_child(round_number, attempt, candidate, groups)
                self.writer.reset()
                self.tracker.reset()
                self._record()
        # Refutations come after all W exact values, and the only evidence is
        # a null-window fail-low in the last line's accepted pass. Its window
        # is that line's best-so-far at the moment the statement is written,
        # not necessarily the W-th value.
        last_attempt = accepted[max(accepted)]
        for candidate in _candidate_groups(
            self.sidecar, round_number, last_attempt
        ).get((), []):
            if candidate.move in last_exact:
                continue
            if candidate.refuted is None:
                self.omitted["bound"] += 1
                continue
            self._write_move(candidate.move, spine=False)
            self.writer.refuted()
            self.refutations += 1
            self.current_refutations += 1
            self._record()
            self.writer.reset()
            self.tracker.reset()
            self._record()

    def synthesize(self):
        self._recorded_root_candidates = []
        for round_start, round_end in zip(
            self.sidecar.round_starts, self.sidecar.round_ends
        ):
            round_number = round_start["round"]
            self.writer.depth()
            self.state.begin_round(round_number)
            self.plies_since_board = 0
            self.current_refutations = 0
            self._record()
            self._recorded_root_candidates = []
            if self.sidecar.width == 1:
                groups = _candidate_groups(
                    self.sidecar, round_number, round_end["attempt"]
                )
                self._write_children(
                    round_number,
                    round_end["attempt"],
                    (),
                    groups.get((), []),
                    groups,
                )
            else:
                self._write_wide_round(round_number)
            top = round_end["rank"][0]
            for entry in round_end["rank"]:
                if entry["move"] not in self.state.current_round:
                    raise SynthesisError(
                        "a rank entry has no written exact root candidate"
                    )
            # The engine's rank is the authoritative order (docs/teacher.md).
            # Equal root values are possible, and the engine breaks the tie with
            # a previous-iteration score that the trace does not carry, so the
            # written candidates can only be required to agree on values: the
            # champion is a written maximum, and nothing written exceeds it.
            if self.state.current_round[top["move"]][0] != top["raw"]:
                raise SynthesisError(
                    "the written value of the round champion differs from the rank"
                )
            written_max = max(value[0] for value in self.state.current_round.values())
            if written_max > top["raw"]:
                raise SynthesisError(
                    "a written root candidate exceeds the round champion"
                )
            raw, wdl = top["raw"], top.get("wdl")
            token = _thought_token(
                raw, wdl, 0, self.sidecar.value_mate, self.sidecar.max_ply
            )
            self.state.finish_round(top["move"], raw, token)
            self._write_eval(token)
            self.round_audits.append(
                RoundAudit(
                    round_number,
                    round_end["depth"],
                    top["move"],
                    raw,
                    None if wdl is None else tuple(wdl),
                    None
                    if wdl is not None
                    else mate_distance_moves(
                        raw, self.sidecar.value_mate, self.sidecar.max_ply
                    ),
                    token,
                    tuple(top["pv"]),
                    tuple(self._recorded_root_candidates),
                    self.current_refutations,
                )
            )
        self._write_final()
        return self._present()

    def _write_final(self):
        top = self.sidecar.round_ends[-1]["rank"][0]
        claimable = (
            self.tracker.root_repetition >= 3
            or chess.Board(self.sidecar.fen).halfmove_clock >= 100
        )
        claim = (
            self.config.synthesis.claim_draw
            and claimable
            and top["raw"] <= DRAW_CLAIM_BOUND
        )
        if claim:
            self.writer.final(CLAIM_DRAW_UCI)
        else:
            self.writer.final(top["move"])
        self.writer.final_evaluate(
            **_final_parameters(
                top["raw"],
                top.get("wdl"),
                self.sidecar.value_mate,
                self.sidecar.max_ply,
            )
        )
        if not claim:
            for move in top["pv"][1:]:
                try:
                    self.writer.final_pv(move)
                except ValueError as error:
                    raise SynthesisError(
                        f"the deepest principal variation is not writable: {move}"
                    ) from error
        self.writer.end()
        self._final_top = top
        self._claim = claim

    def _present(self):
        tokens = tuple(int(token) for token in self.writer.tokens)
        roles = tuple(int(role) for role in self.writer.roles)
        final_step = next(
            step for step in self.writer.steps if step.operation == "final"
        )
        suffix_start = final_step.start
        suffix_tokens = tokens[suffix_start:]
        suffix_roles = roles[suffix_start:]
        if suffix_tokens[0] != FINAL:
            raise SynthesisError("the label suffix does not start with FINAL")
        final_move = self.writer.final_move
        value_token = suffix_tokens[2]
        mate_distance = None
        if 3 < len(suffix_tokens) and MATE_WIN <= suffix_tokens[3] < PAD:
            mate_token_id = suffix_tokens[3]
            mate_distance = (
                -(mate_token_id - MATE_LOSS)
                if mate_token_id >= MATE_LOSS
                else mate_token_id - MATE_WIN
            )
        rows = {}
        for presentation in PRESENTATIONS:
            if presentation == "zero":
                cut = 0
                boundaries = ()
            elif presentation == "full":
                cut = suffix_start
                boundaries = tuple(self.boundaries)
            elif presentation == "random":
                random_rng = random.Random(f"{self.seed}:{self.root['root_id']}:random")
                low = math.ceil(
                    self.config.presentation.random_min_fraction * suffix_start
                )
                high = math.floor(
                    self.config.presentation.random_max_fraction * suffix_start
                )
                low = max(1, min(low, suffix_start))
                high = max(low, min(high, suffix_start))
                target = random_rng.randint(low, high)
                index = next(
                    (
                        position
                        for position, boundary in enumerate(self.boundaries)
                        if boundary.tool_end >= target
                    ),
                    len(self.boundaries) - 1,
                )
                cut = self.boundaries[index].tool_end
                boundaries = tuple(self.boundaries[: index + 1])
            elif presentation == "last_root":
                if len(self.boundaries) < 2:
                    raise SynthesisError("no boundary before the final root evaluation")
                cut = self.boundaries[-2].tool_end
                boundaries = tuple(self.boundaries[:-1])
            else:
                raise AssertionError(presentation)
            prefix = tokens[:cut]
            rows[presentation] = PresentationRow(
                presentation,
                prefix + suffix_tokens,
                roles[:cut] + suffix_roles,
                boundaries,
                sum(1 for token in prefix if token == DEPTH),
                cut,
                final_move,
                value_token - VALUE,
                mate_distance,
                final_move == CLAIM_DRAW_UCI,
            )
        stats = self._stats(rows)
        return SynthesizedRoot(self.root, rows, tuple(self.round_audits), stats)

    def _stats(self, rows):
        audits = self.round_audits
        champions = [audit.move for audit in audits]
        changes = [
            index
            for index in range(1, len(champions))
            if champions[index] != champions[index - 1]
        ]
        last_change = (changes[-1] + 1) if changes else 1
        root_values = [
            None if audit.wdl is None else (audit.wdl[0] - audit.wdl[2]) / 1000
            for audit in audits
        ]
        all_draw = all(
            audit.wdl is not None and audit.wdl == (0, 1000, 0) and audit.mate is None
            for audit in audits
        )
        return {
            "rounds": len(audits),
            "depth": audits[-1].depth,
            "teacher_width": self.sidecar.configured_width,
            "champions": champions,
            "overturned_rounds": len(changes),
            "overturn_rate": len(changes) / max(len(audits) - 1, 1),
            "last_change_round": last_change,
            "converged_depth": audits[last_change - 1].depth,
            "stable_tail_rounds": len(audits) - last_change,
            "first_round_matches_final": champions[0] == champions[-1],
            "refutations": self.refutations,
            "refutations_per_round": [audit.refutations for audit in audits],
            "root_values": root_values,
            "root_mates": [audit.mate for audit in audits],
            "all_draw": all_draw,
            "claim_draw": rows["full"].claim_draw,
            "claimable": self.tracker.root_repetition >= 3
            or chess.Board(self.sidecar.fen).halfmove_clock >= 100,
            "thinking_tokens": rows["full"].thinking_tokens,
            "total_tokens": len(rows["full"].tokens),
            "board_descent": self.board_counts["descent"],
            "board_frontier": self.board_counts["frontier"],
            "exact_evals": self.exact_evals,
            "terminal_evals": self.terminal_evals,
            "frontier_evals": self.frontier_evals,
            "omitted_bound": self.omitted["bound"],
            "omitted_early_refutation": self.omitted["early_refutation"],
            "omitted_root_fail_high": self.omitted["root_fail_high"],
        }


def validate_root(root, sidecar):
    """Check root metadata against its sidecar and the collection contract."""
    if root.get("fen") != sidecar.fen:
        raise SynthesisError("root metadata FEN differs from the sidecar")
    if root.get("root_id") is None:
        raise SynthesisError("root metadata has no root_id")
    if root.get("split") not in {"train", "val"}:
        raise SynthesisError("root split must be train or val")
    repetition = _number(root.get("repetition"), "root.repetition", 1, 4)
    board = chess.Board(sidecar.fen)
    expected_turn = "white" if board.turn == chess.WHITE else "black"
    if root.get("turn") != expected_turn:
        raise SynthesisError("root turn differs from the sidecar FEN")
    if root.get("position_key") is not None:
        key = " ".join(sidecar.fen.split()[:4])
        if root["position_key"] != key:
            raise SynthesisError("root position_key differs from the sidecar FEN")
    if repetition != int(root["repetition"]):
        raise SynthesisError("root repetition is invalid")
    if "teacher_width" in root:
        width = _number(root["teacher_width"], "root.teacher_width", 1)
        if width != sidecar.configured_width:
            raise SynthesisError("root teacher width differs from the sidecar")
    if "depth" in root:
        depth = _number(root["depth"], "root.depth", 1)
        if depth != sidecar.search_start["depth"]:
            raise SynthesisError("root depth differs from the sidecar")
    return repetition


def synthesize_root(sidecar, root, config, seed=None):
    """Synthesize one root, its four presentation rows, and its audit trail."""
    validate_root(root, sidecar)
    if sidecar.search_start["depth"] != sidecar.round_ends[-1]["depth"]:
        raise SynthesisError("the recorded search did not complete its requested depth")
    return _Synthesis(sidecar, root, config, seed).synthesize()


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
        stats = root.stats
        selected = self.presentations[root.root["root_id"]]
        self.roots += 1
        self.rows += len(selected)
        self.transitions += max(stats["rounds"] - 1, 0)
        self.overturns += stats["overturned_rounds"]
        self.refutations += stats["refutations"]
        self.rounds += stats["rounds"]
        self.full_lengths.append(stats["thinking_tokens"])
        self.total_lengths.append(stats["total_tokens"])
        for name in selected:
            row = root.rows[name]
            self.presentation_lengths[name].append(len(row.tokens))
            self.presentation_thinking[name].append(row.thinking_tokens)
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
        self.splits[root.root["split"]] += 1
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
                "root_id": root.root["root_id"],
                "split": root.root["split"],
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
