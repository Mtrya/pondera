"""Parse and validate one instrumented Stockfish search record.

The sidecar is the only source of teacher statements. This module checks the
record against the schema described in docs/data.md and exposes the indexed events;
it does not translate them into trace tokens.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import chess

from pondera.chess.tracker import PathTracker

SIDECAR_SCHEMA = 3
SIDECAR_SCHEMA_WIDE = 4  # MultiPV teacher: per-line attempts (docs/data.md)
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


def require_int(value, name, minimum=None, maximum=None):
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
    alpha = require_int(value["alpha"], f"{name}.alpha")
    beta = require_int(value["beta"], f"{name}.beta")
    if alpha > beta:
        raise SynthesisError(f"{name} has alpha above beta")
    return alpha, beta


def is_mate_score(raw, value_mate, max_ply):
    return abs(raw) >= value_mate - max_ply


def _check_cp_wdl(event, value_mate, max_ply, name):
    raw = require_int(event["raw"], f"{name}.raw")
    mate = is_mate_score(raw, value_mate, max_ply)
    if mate:
        if "cp" in event or "wdl" in event:
            raise SynthesisError(f"{name} has cp/wdl on a mate score")
        return raw, None, None
    if "cp" not in event or "wdl" not in event:
        raise SynthesisError(f"{name} is missing cp/wdl on an ordinary score")
    cp = require_int(event["cp"], f"{name}.cp")
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
    value_mate = require_int(session.get("value_mate"), "session.value_mate", 1)
    max_ply = require_int(session.get("max_ply"), "session.max_ply", 1)
    options = session.get("options")
    if not isinstance(options, dict) or set(options) != {
        "threads",
        "hash_mb",
        "multipv",
    }:
        raise SynthesisError("session.options is incomplete")
    require_int(options["threads"], "session.options.threads", 1)
    require_int(options["hash_mb"], "session.options.hash_mb", 1)
    require_int(options["multipv"], "session.options.multipv", 1)
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
    depth = require_int(search_start.get("depth"), "search_start.depth", 1)
    require_int(search_start.get("multi_pv"), "search_start.multi_pv", 1)
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
        require_int(start.get("round"), "round_start.round", 1)
        require_int(start.get("depth"), "round_start.depth", 1)
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
        round_number = require_int(event.get("round"), "attempt.round", 1)
        if round_number not in attempts_by_round:
            raise SynthesisError("attempt references an unknown round")
        require_int(event.get("attempt"), "attempt.attempt", 1)
        line = event.get("line")
        if configured_wide:
            if line is None:
                raise SynthesisError("a wide-teacher attempt has no line")
            require_int(line, "attempt.line", 0, effective_width - 1)
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
        require_int(end_event.get("attempt"), "round_end.attempt", 1)
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
        round_number = require_int(event.get("round"), f"{kind}.round", 1)
        if round_number not in attempts_by_round:
            raise SynthesisError(f"{kind} references an unknown round")
        attempt_number = require_int(event.get("attempt"), f"{kind}.attempt", 1)
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
            require_int(event.get("order"), "candidate.order", 1)
            call = event.get("call")
            if call not in CALL_ROLES:
                raise SynthesisError("candidate.call is unknown")
            require_int(event.get("depth"), "candidate.depth")
            alpha, beta = _window(event.get("window"), "candidate.window")
            if call in {"probe", "rescore"} and beta - alpha != 1:
                raise SynthesisError("a probe or rescore must use a null window")
            raw, cp, wdl = _check_cp_wdl(event, value_mate, max_ply, "candidate")
            bound = event.get("bound")
            if bound not in BOUNDS or bound != _bound_of(raw, alpha, beta):
                raise SynthesisError(
                    "candidate.bound disagrees with its window and value"
                )
            require_int(event.get("nodes_delta"), "candidate.nodes_delta", 0)
            require_int(event.get("seldepth"), "candidate.seldepth", 0)
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
            require_int(event.get("nodes_delta"), "frontier.nodes_delta", 0)
            require_int(event.get("seldepth"), "frontier.seldepth", 0)

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
            # is not the exact mirror of the root wdl (docs/data.md).

    if search_end.get("bestmove") is None:
        raise SynthesisError("search_end has no bestmove")
    if search_end["bestmove"] != round_ends[-1]["rank"][0]["move"]:
        raise SynthesisError("search_end.bestmove differs from the deepest rank[0]")
    require_int(search_end.get("nodes"), "search_end.nodes", 0)
    require_int(search_end.get("seldepth"), "search_end.seldepth", 0)

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
