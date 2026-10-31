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
from pondera.data.trace import CLAIM_DRAW_UCI, TraceWriter

SIDECAR_SCHEMA = 3
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
    """A validated schema-3 record, indexed by event type."""

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
    """Parse and independently validate one sidecar JSONL stream."""
    if teacher_width != 1:
        raise SynthesisError("sidecar translation is defined for teacher width one")
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
        if event.get("schema") != SIDECAR_SCHEMA:
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
    depth = _number(search_start.get("depth"), "search_start.depth", 1)
    if search_start.get("multi_pv") != teacher_width:
        raise SynthesisError("search_start.multi_pv differs from the teacher width")

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
    for event in attempts:
        round_number = _number(event.get("round"), "attempt.round", 1)
        if round_number not in attempts_by_round:
            raise SynthesisError("attempt references an unknown round")
        _number(event.get("attempt"), "attempt.attempt", 1)
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
        exact = [event for event in round_attempts if event["bound"] == "exact"]
        if len(exact) != 1:
            raise SynthesisError(
                f"round {round_number} has {len(exact)} exact attempts"
            )
        if end_event.get("attempt") != exact[0]["attempt"]:
            raise SynthesisError(
                f"round {round_number} does not accept its exact attempt"
            )
        _number(end_event.get("attempt"), "round_end.attempt", 1)
        rank = end_event.get("rank")
        if not isinstance(rank, list) or not rank or len(rank) > teacher_width:
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
        accepted = exact[0]
        if top_raw != accepted["raw"]:
            raise SynthesisError("rank[0].raw differs from the accepted attempt")
        if "cp" in accepted and top_cp != accepted["cp"]:
            raise SynthesisError("rank[0].cp differs from the accepted attempt")
        if "wdl" in accepted and list(top_wdl) != list(accepted["wdl"]):
            raise SynthesisError("rank[0].wdl differs from the accepted attempt")

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
        accepted = end_event["attempt"]
        groups = {}
        for event in candidates:
            if event["round"] == round_number and event["attempt"] == accepted:
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

    # The accepted root call establishes the round value exactly.
    for end_event in round_ends:
        round_number = end_event["round"]
        accepted = end_event["attempt"]
        top = end_event["rank"][0]
        establishing = [
            event
            for event in candidates
            if event["round"] == round_number
            and event["attempt"] == accepted
            and event["ply"] == 1
            and event["move"] == top["move"]
            and event["call"] in {"first", "promote"}
            and event["bound"] == "exact"
        ]
        if not establishing:
            raise SynthesisError("the root establishing call is not exact")
        call = establishing[-1]
        if call["raw"] != -top["raw"]:
            raise SynthesisError("the root establishing call does not negate rank[0]")
        if call.get("pv") != top.get("pv"):
            raise SynthesisError("the root establishing call pv differs from rank[0]")
        if "wdl" in top and "wdl" in call:
            if list(call["wdl"]) != [top["wdl"][2], top["wdl"][1], top["wdl"][0]]:
                raise SynthesisError(
                    "the root establishing wdl is not the rank[wdl] mirror"
                )

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


def load_sidecar(path, teacher_width=1):
    path = Path(path)
    return parse_sidecar(
        path.read_text(encoding="utf-8").splitlines(),
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
        self.previous = None
        self.current = None

    def begin_round(self, round_number):
        self.round = round_number
        self.current_round = {}
        self.current = self.previous

    def record_exact(self, move, raw, token, order):
        self.current_round[move] = (raw, token, order)
        candidates = [
            (raw, -order, move, token)
            for move, (raw, token, order) in self.current_round.items()
        ]
        best_raw, _, best_move, best_token = max(candidates)
        self.current = (_move_token(best_move, self.mirrored), best_token, best_raw)

    def finish_round(self, move, raw, token):
        self.previous = (_move_token(move, self.mirrored), token, raw)
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
            teacher_width=config.search.teacher_width,
        )
        self.tracker = PathTracker(sidecar.fen, int(root["repetition"]))
        self.state = _RootState(self.writer.mirrored, config.search.teacher_width)
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

    def _record_root_candidate(self, round_number, candidate, token):
        if candidate.parent != ():
            return
        raw = -candidate.exact["raw"]
        self.state.record_exact(candidate.move, raw, token, candidate.order)
        wdl = candidate.exact.get("wdl")
        self._recorded_root_candidates.append(
            CandidateAudit(
                round_number,
                candidate.move,
                raw,
                None if wdl is None else (wdl[2], wdl[1], wdl[0]),
                _token_mate(token),
                token,
                candidate.order,
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
                if raw != 0:
                    raise SynthesisError(
                        "a rule draw child has a non-draw engine value"
                    )
                token = VALUE + value_bin(0.0)
            self.terminal_evals += 1
            self._record_root_candidate(round_number, candidate, token)
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
        needed = self.config.search.teacher_width if parent == () else 1
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
            groups = _candidate_groups(self.sidecar, round_number, round_end["attempt"])
            self._recorded_root_candidates = []
            self._write_children(
                round_number,
                round_end["attempt"],
                (),
                groups.get((), []),
                groups,
            )
            top = round_end["rank"][0]
            if top["move"] not in self.state.current_round:
                raise SynthesisError(
                    "the round champion has no written exact root candidate"
                )
            best_move = max(
                self.state.current_round.items(),
                key=lambda item: (item[1][0], -item[1][2]),
            )[0]
            if best_move != top["move"]:
                raise SynthesisError(
                    "the written root candidates do not reproduce the round champion"
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
        claim = self.config.synthesis.claim_draw and claimable and top["raw"] <= 0
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


def validate_root(root, sidecar, teacher_width=1):
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
    if teacher_width != 1:
        raise SynthesisError("teacher width one is required")
    return repetition


def synthesize_root(sidecar, root, config, seed=None):
    """Synthesize one root, its four presentation rows, and its audit trail."""
    validate_root(root, sidecar, config.search.teacher_width)
    if sidecar.search_start["depth"] != sidecar.round_ends[-1]["depth"]:
        raise SynthesisError("the recorded search did not complete its requested depth")
    if sidecar.search_start["depth"] != config.search.depth:
        raise SynthesisError(
            "the sidecar depth differs from the configured collection depth"
        )
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


def profile_roots(roots):
    """Aggregate the pilot statistics over synthesized roots."""
    roots = list(roots)
    if not roots:
        raise ValueError("A statistics profile needs at least one root")
    transitions = sum(max(stats["rounds"] - 1, 0) for stats in (r.stats for r in roots))
    overturns = sum(stat["overturned_rounds"] for stat in (r.stats for r in roots))
    refutations = sum(stat["refutations"] for stat in (r.stats for r in roots))
    rounds = sum(stat["rounds"] for stat in (r.stats for r in roots))
    full_lengths = [r.stats["thinking_tokens"] for r in roots]
    total_lengths = [r.stats["total_tokens"] for r in roots]
    presentation_lengths = {
        name: [len(root.rows[name].tokens) for root in roots] for name in PRESENTATIONS
    }
    presentation_thinking = {
        name: [root.rows[name].thinking_tokens for root in roots]
        for name in PRESENTATIONS
    }
    rounds_distribution = Counter(r.stats["rounds"] for r in roots)
    converged_distribution = Counter(r.stats["converged_depth"] for r in roots)
    refutation_distribution = Counter()
    for root in roots:
        for count in root.stats["refutations_per_round"]:
            refutation_distribution[count] += 1
    value_shapes = Counter()
    for root in roots:
        values = [value for value in root.stats["root_values"] if value is not None]
        if not values:
            value_shapes["mate_only"] += 1
            continue
        changes = sum(1 for left, right in zip(values, values[1:]) if left != right)
        if changes == 0:
            value_shapes["flat"] += 1
        elif all(right >= left for left, right in zip(values, values[1:])):
            value_shapes["monotone_up"] += 1
        elif all(right <= left for left, right in zip(values, values[1:])):
            value_shapes["monotone_down"] += 1
        else:
            value_shapes["mixed"] += 1
    splits = Counter(root.root["split"] for root in roots)
    return {
        "roots": len(roots),
        "rows": len(roots) * len(PRESENTATIONS),
        "split_counts": dict(sorted(splits.items())),
        "presentation_counts": {name: len(roots) for name in PRESENTATIONS},
        "rounds": dict(sorted(rounds_distribution.items())),
        "rounds_summary": _percentiles([r.stats["rounds"] for r in roots]),
        "thinking_tokens": _percentiles(full_lengths),
        "total_tokens": _percentiles(total_lengths),
        "presentation_lengths": {
            name: _percentiles(values) for name, values in presentation_lengths.items()
        },
        "presentation_thinking_tokens": {
            name: _percentiles(values) for name, values in presentation_thinking.items()
        },
        "overturned_rounds": overturns,
        "overturn_rate": overturns / transitions if transitions else 0.0,
        "roots_with_overturn": sum(
            1 for root in roots if root.stats["overturned_rounds"] > 0
        ),
        "refutations": refutations,
        "refutations_per_round": refutations / max(rounds, 1),
        "refutations_per_round_distribution": dict(
            sorted(refutation_distribution.items())
        ),
        "converged_depth": dict(sorted(converged_distribution.items())),
        "first_round_matches_final": sum(
            1 for root in roots if root.stats["first_round_matches_final"]
        ),
        "value_trajectory_shapes": dict(sorted(value_shapes.items())),
        "all_draw_roots": sum(1 for root in roots if root.stats["all_draw"]),
        "claim_draw_roots": sum(1 for root in roots if root.stats["claim_draw"]),
        "claimable_roots": sum(1 for root in roots if root.stats["claimable"]),
        "board_descent": sum(root.stats["board_descent"] for root in roots),
        "board_frontier": sum(root.stats["board_frontier"] for root in roots),
        "exact_evals": sum(root.stats["exact_evals"] for root in roots),
        "terminal_evals": sum(root.stats["terminal_evals"] for root in roots),
        "frontier_evals": sum(root.stats["frontier_evals"] for root in roots),
        "omitted_bound": sum(root.stats["omitted_bound"] for root in roots),
        "omitted_early_refutation": sum(
            root.stats["omitted_early_refutation"] for root in roots
        ),
        "omitted_root_fail_high": sum(
            root.stats["omitted_root_fail_high"] for root in roots
        ),
        "per_root": [
            {
                "root_id": root.root["root_id"],
                "split": root.root["split"],
                "rounds": root.stats["rounds"],
                "thinking_tokens": root.stats["thinking_tokens"],
                "overturned_rounds": root.stats["overturned_rounds"],
                "all_draw": root.stats["all_draw"],
                "claim_draw": root.stats["claim_draw"],
            }
            for root in roots
        ],
    }
