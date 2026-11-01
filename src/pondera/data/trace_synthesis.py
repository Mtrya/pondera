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

import math
import random
from dataclasses import dataclass

import chess

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
from pondera.data.sidecar import SynthesisError, is_mate_score, require_int
from pondera.data.trace import CLAIM_DRAW_UCI, DRAW_CLAIM_BOUND, TraceWriter

PRESENTATIONS = ("zero", "full", "random", "last_root")


def mate_distance_moves(raw, value_mate, max_ply):
    """Convert a root-player mate score from plies to UCI move distance.

    This is the ``UCIEngine::format_score`` rule: a win rounds the remaining
    plies up by one before halving; a loss truncates toward zero before the
    sign is applied.
    """
    if not is_mate_score(raw, value_mate, max_ply):
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
                    not is_mate_score(
                        raw, self.sidecar.value_mate, self.sidecar.max_ply
                    )
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
            # The engine's rank is the authoritative order (docs/data.md).
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
    repetition = require_int(root.get("repetition"), "root.repetition", 1, 4)
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
        width = require_int(root["teacher_width"], "root.teacher_width", 1)
        if width != sidecar.configured_width:
            raise SynthesisError("root teacher width differs from the sidecar")
    if "depth" in root:
        depth = require_int(root["depth"], "root.depth", 1)
        if depth != sidecar.search_start["depth"]:
            raise SynthesisError("root depth differs from the sidecar")
    return repetition


def synthesize_root(sidecar, root, config, seed=None):
    """Synthesize one root, its four presentation rows, and its audit trail."""
    validate_root(root, sidecar)
    if sidecar.search_start["depth"] != sidecar.round_ends[-1]["depth"]:
        raise SynthesisError("the recorded search did not complete its requested depth")
    return _Synthesis(sidecar, root, config, seed).synthesize()
