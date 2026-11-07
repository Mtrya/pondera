"""Single-root trace grammar, teacher-record validation, and synthesis.

The module is deliberately free of pipeline, filesystem-layout, and training
concerns. ``TraceWriter`` and ``replay`` implement the grammar; ``parse_sidecar``
provides exhaustive producer diagnostics. Normal curation decodes qualified
producer output and synthesizes presentation rows without replaying the search.
"""

import json
import math
import random
from bisect import bisect_right
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import NamedTuple

import chess
import numpy as np
import orjson

from pondera.chess.codec import (
    decode_board,
    decode_value_token,
    mate_token,
    value_bin,
)
from pondera.chess.symmetry import TOKEN_MIRROR, reflect_uci
from pondera.chess.tracker import PathTracker
from pondera.chess.vocabulary import (
    BACK,
    BOARD,
    DEPTH,
    END,
    FINAL,
    IDX_TO_UCI_MOVE,
    MATE_LOSS,
    MATE_WIN,
    META_END,
    MOVE,
    PAD,
    REFUTED,
    RESET,
    SIDE,
    UCI_MOVE_TO_IDX,
    VALUE,
    VALUE_BINS,
)

# The grammar version recorded in trace dataset manifests.
TRACE_GRAMMAR_VERSION = 2

# Token roles support masked losses and per-segment diagnostics.
ROLE_PAD = 0
ROLE_BOARD = 1
ROLE_MOVE = 2
ROLE_EVAL = 3
ROLE_CONTROL = 4
ROLE_FINAL_MOVE = 5
ROLE_REFUTED = 6
ROLE_FINAL_PV = 7
ROLE_FINAL_EVAL = 8
ROLE_COUNT = 9

# One board observation carries eight metadata atoms after its piece atoms.
OBSERVATION_METADATA = 8

CLAIM_DRAW_UCI = "<claim_draw>"
CLAIM_DRAW = MOVE + UCI_MOVE_TO_IDX[CLAIM_DRAW_UCI]

# The engine's rule-draw score is VALUE_DRAW - 1 + (nodes & 0x2) ∈ {-1, +1},
# an anti-cycling jitter rather than an evaluation, so the draw-claim gate
# treats raw values up to this bound as "not better than a draw".
DRAW_CLAIM_BOUND = 1

_THOUGHT = "thought"
_SUFFIX = "suffix"
_DONE = "done"

# Suffix sub-states: awaiting the decision move, then the mandatory value
# token, then free continuation through a mate token, a best line, and END.
_DECISION = "decision"
_VALUE = "value"
_MATE = "mate"
_PV = "pv"


class TraceError(ValueError):
    """A malformed or rule-inconsistent trace, tagged at its first bad token."""

    def __init__(self, tag, index, detail=""):
        self.tag = tag
        self.index = index
        message = f"{tag} at token {index}"
        if detail:
            message += f": {detail}"
        super().__init__(message)


class _Illegal(ValueError):
    """A state-machine rejection, carrying the replay tag for the first token."""

    def __init__(self, tag, detail=""):
        self.tag = tag
        self.detail = detail
        super().__init__(detail or tag)


def is_move_token(token):
    return MOVE <= token < VALUE


def is_eval_token(token):
    return VALUE <= token < PAD


def is_metadata_token(token):
    return SIDE <= token <= META_END


def derive_trace_roles(tokens):
    """Recover token roles from grammar delimiters without chess replay."""
    tokens = [int(token) for token in tokens]
    roles = []
    index = 0
    suffix = False
    while index < len(tokens):
        token = tokens[index]
        if not suffix and token == BOARD:
            roles.append(ROLE_CONTROL)
            index += 1
            start = index
            while index < len(tokens) and tokens[index] < MOVE:
                roles.append(ROLE_BOARD)
                index += 1
            observation = tokens[start:index]
            if (
                len(observation) < 9
                or any(token >= SIDE for token in observation[:-8])
                or any(token < SIDE for token in observation[-8:])
            ):
                raise ValueError("A BOARD operation has an invalid observation")
            continue
        if not suffix:
            if token == FINAL:
                suffix = True
                roles.append(ROLE_CONTROL)
            elif token == REFUTED:
                roles.append(ROLE_REFUTED)
            elif is_move_token(token):
                roles.append(ROLE_MOVE)
            elif is_eval_token(token):
                roles.append(ROLE_EVAL)
            elif token in {BACK, RESET, DEPTH}:
                roles.append(ROLE_CONTROL)
            else:
                raise ValueError(f"Unexpected thought token {token} at index {index}")
        elif index > 0 and tokens[index - 1] == FINAL:
            if not is_move_token(token):
                raise ValueError("FINAL must be followed by a decision")
            roles.append(ROLE_FINAL_MOVE)
        elif token == END:
            roles.append(ROLE_CONTROL)
        elif is_move_token(token):
            roles.append(ROLE_FINAL_PV)
        elif is_eval_token(token):
            roles.append(ROLE_FINAL_EVAL)
        else:
            raise ValueError(f"Unexpected final token {token} at index {index}")
        index += 1
    return np.asarray(roles, dtype=np.int8)


@dataclass(frozen=True)
class Step:
    """One validated trace operation with its token span and resulting state.

    ``position`` is the absolute current node after the operation. ``observed``
    is the decoded content of a BOARD response. ``mate`` is a nonnegative
    distance and ``mate_win`` says whose mate it is; ``evaluation`` is the
    token's decoded scalar. A ``final`` step keeps the thought path it ended;
    ``final_evaluate``, ``final_pv``, and ``end`` describe the label suffix,
    whose path starts again at the root.
    """

    start: int
    end: int
    operation: str
    path: tuple
    position: str
    uci: str | None = None
    san: str | None = None
    evaluation: float | None = None
    mate: int | None = None
    mate_win: bool | None = None
    observed: str | None = None
    repetition: int | None = None


@dataclass(frozen=True)
class TraceProgram:
    """A replayed trace: tokens, roles, steps, and the root decision."""

    root_fen: str
    root_repetition: int
    tokens: tuple
    roles: tuple
    steps: tuple
    final_move: str
    mirrored: bool

    @property
    def final_move_index(self):
        """Token index of the decision after FINAL; it is not the trace end."""
        for step in self.steps:
            if step.operation == "final":
                return step.start + 1
        raise ValueError("Replayed trace has no final decision")

    def render(self):
        """Natural-language transcription of the replayed program."""
        lines = [f"Root position (repetition {self.root_repetition}): {self.root_fen}"]
        if self.mirrored:
            lines.append(
                "Black-to-move root: trace moves and observations are rank-mirrored."
            )
        for number, step in enumerate(self.steps, 1):
            span = f"[{step.start}:{step.end}]"
            where = " ".join(step.path) or "root"
            if step.operation == "move":
                lines.append(
                    f"{number}. {span} move {step.san} ({step.uci}); "
                    f"path {where}; position {step.position}"
                )
            elif step.operation == "back":
                lines.append(
                    f"{number}. {span} BACK; path {where}; position {step.position}"
                )
            elif step.operation == "reset":
                lines.append(
                    f"{number}. {span} RESET; path {where}; position {step.position}"
                )
            elif step.operation == "depth":
                lines.append(f"{number}. {span} <depth> round {number}")
            elif step.operation == "refuted":
                lines.append(
                    f"{number}. {span} <refuted>; path {where}; position {step.position}"
                )
            elif step.operation == "evaluate":
                if step.mate is not None:
                    verdict = (
                        f"mate in {step.mate} "
                        f"({'win' if step.mate_win else 'loss'}) for the root player"
                    )
                else:
                    verdict = f"value {step.evaluation:+.3f} for the root player"
                lines.append(
                    f"{number}. {span} evaluate {verdict}; "
                    f"path {where}; position {step.position}"
                )
            elif step.operation == "board":
                lines.append(
                    f"{number}. {span} BOARD; observed {step.observed} "
                    f"(repetition {step.repetition}); path {where}"
                )
            elif step.operation == "final":
                decision = (
                    "claim draw"
                    if step.uci == CLAIM_DRAW_UCI
                    else f"{step.san} ({step.uci})"
                )
                lines.append(
                    f"{number}. {span} FINAL {decision} from the root; "
                    f"thought ended at path {where}; position {step.position}"
                )
            elif step.operation == "final_evaluate":
                if step.mate is not None:
                    verdict = (
                        f"mate in {step.mate} "
                        f"({'win' if step.mate_win else 'loss'}) for the root player"
                    )
                else:
                    verdict = f"value {step.evaluation:+.3f} for the root player"
                lines.append(f"{number}. {span} FINAL evaluation {verdict}")
            elif step.operation == "final_pv":
                lines.append(
                    f"{number}. {span} FINAL line {step.san} ({step.uci}); "
                    f"path {where}; position {step.position}"
                )
            elif step.operation == "end":
                lines.append(f"{number}. {span} <end>")
        return "\n".join(lines)


def _board_move(uci):
    """Parse an absolute UCI board move, rejecting the draw-claim pseudo-move."""
    try:
        return chess.Move.from_uci(uci)
    except ValueError as error:
        raise ValueError(f"Not a board move: {uci}") from error


def _absolute_uci(token, mirrored):
    uci = IDX_TO_UCI_MOVE[token - MOVE]
    if uci == CLAIM_DRAW_UCI:
        return uci
    return reflect_uci(uci) if mirrored else uci


def _decode_observation(tokens, mirrored):
    """Decode a stored observation into the absolute FEN and repetition."""
    values = (
        [int(TOKEN_MIRROR[token]) for token in tokens] if mirrored else list(tokens)
    )
    return decode_board(values)


def _read_observation(values, request):
    """Read one BOARD response starting after the request token."""
    start = request + 1
    index = start
    while index < len(values) and 0 <= values[index] < SIDE:
        index += 1
    if index == start:
        raise TraceError("board_truncated", request, "missing piece atoms")
    end = index + OBSERVATION_METADATA
    if end > len(values) or not all(
        is_metadata_token(values[position]) for position in range(index, end)
    ):
        raise TraceError("board_truncated", request, "missing metadata atoms")
    return values[start:end], end


def _mate_fields(token):
    if MATE_WIN <= token < MATE_LOSS:
        return token - MATE_WIN, True
    if MATE_LOSS <= token < PAD:
        return token - MATE_LOSS, False
    return None, None


class _Visit:
    """Outcomes of one node visit inside one round."""

    __slots__ = (
        "node",
        "exact",
        "refuted_children",
        "passed_children",
        "explored",
        "evaluated",
        "refuted",
        "written",
    )

    def __init__(self, node):
        self.node = node
        self.exact = {}
        self.refuted_children = set()
        self.passed_children = set()
        self.explored = False
        self.evaluated = False
        self.refuted = False
        self.written = False

    @property
    def concluded(self):
        return self.evaluated or self.refuted


class _TraceState:
    """The one grammar state machine behind the writer, replay, and the mask."""

    def __init__(self, root_fen, repetition=1, teacher_width=1):
        if teacher_width < 1:
            raise ValueError("Teacher width must be positive")
        self.tracker = PathTracker(root_fen, repetition)
        self.teacher_width = teacher_width
        self.root_fen = root_fen
        self.root = self.tracker.board.copy()
        self.mirrored = self.root.turn == chess.BLACK
        self.root_position = self.root.fen(en_passant="fen")
        self.root_terminal = self.tracker.search_terminal()
        self.tokens = []
        self.roles = []
        self.steps = []
        self.round = 0
        self.round_has_candidate = False
        self.visits = []
        self.pending_board = None
        self.phase = _THOUGHT
        self.suffix = None
        self.value_written = False
        self.mate_written = False
        self.value_bin = None
        self.claim = False
        self.pv_tracker = None
        self.final_move = None
        self._final_start = None
        self._final_path = None
        self._final_fen = None
        root_moves = (
            ()
            if self.root_terminal is not None
            else tuple(
                sorted(self._move_token(move.uci()) for move in self.root.legal_moves)
            )
        )
        self.decision_tokens = root_moves + ((CLAIM_DRAW,) if self._claimable() else ())

    # --- token and step plumbing --- #

    def _emit(self, token, role):
        self.tokens.append(int(token))
        self.roles.append(role)

    def _step(self, start, operation, path, position, **fields):
        self.steps.append(
            Step(start, len(self.tokens), operation, path, position, **fields)
        )

    def _move_token(self, uci):
        if uci == CLAIM_DRAW_UCI:
            return CLAIM_DRAW
        framed = reflect_uci(uci) if self.mirrored else uci
        return MOVE + UCI_MOVE_TO_IDX[framed]

    def _legal_move_tokens(self, tracker, cache):
        path = tracker.path
        if cache is not None and path in cache:
            return cache[path]
        if tracker.search_terminal() is not None:
            moves = ()
        else:
            moves = tuple(self._move_token(uci) for uci in tracker.legal_moves())
        if cache is not None:
            cache[path] = moves
        return moves

    # --- state queries --- #

    def _claimable(self):
        return not self.root.is_checkmate() and (
            self.tracker.root_repetition >= 3 or self.root.halfmove_clock >= 100
        )

    def _reference_exists(self):
        parent = self.visits[-2]
        child = self.tracker.path[-1]
        needed = self.teacher_width if len(self.tracker.path) == 1 else 1
        return sum(1 for move in parent.exact if move != child) >= needed

    def _require_thought(self, tag):
        if self.suffix is not None:
            raise _Illegal(tag)

    def allowed_tokens(self, final_allowed, cache=None):
        """Token ids the current state allows, in the vocabulary's frame."""
        if self.phase == _DONE:
            return ()
        if self.pending_board is not None:
            _, observation, index = self.pending_board
            return (observation[index],)
        if self.suffix is None:
            return self._thought_tokens(final_allowed, cache)
        return self._suffix_tokens(cache)

    def _thought_tokens(self, final_allowed, cache):
        tokens = []
        if not self.visits:
            tokens.append(DEPTH)
            if final_allowed and self.decision_tokens:
                tokens.append(FINAL)
            return tuple(tokens)
        visit = self.visits[-1]
        at_root = not self.tracker.path
        if not visit.concluded:
            tokens.extend(self._legal_move_tokens(self.tracker, cache))
            tokens.extend(range(VALUE, PAD))
            if not at_root and not visit.written and self._reference_exists():
                tokens.append(REFUTED)
        if not at_root:
            tokens.append(BOARD)
            tokens.append(BACK)
        if not at_root or not visit.evaluated:
            tokens.append(RESET)
        if final_allowed and self.decision_tokens:
            tokens.append(FINAL)
        if at_root and self.round_has_candidate:
            tokens.append(DEPTH)
        return tuple(tokens)

    def _suffix_tokens(self, cache):
        if self.suffix == _DECISION:
            return tuple(self.decision_tokens)
        if not self.value_written:
            return tuple(range(VALUE, MATE_WIN))
        tokens = [END]
        if (
            not self.mate_written
            and self.suffix == _VALUE
            and self.value_bin in (0, VALUE_BINS - 1)
        ):
            if self.value_bin == VALUE_BINS - 1:
                tokens.extend(range(MATE_WIN, MATE_LOSS))
            else:
                tokens.extend(range(MATE_LOSS, PAD))
        if not self.claim and self.pv_tracker.search_terminal() is None:
            tokens.extend(self._legal_move_tokens(self.pv_tracker, cache))
        return tuple(tokens)

    # --- thought operations --- #

    def depth(self):
        self._require_thought("depth_after_final")
        if self.tracker.path:
            raise _Illegal("depth_at_depth")
        if self.round and not self.round_has_candidate:
            raise _Illegal("depth_without_candidate")
        start = len(self.tokens)
        self.round += 1
        self.round_has_candidate = False
        self.visits = [_Visit(())]
        self._emit(DEPTH, ROLE_CONTROL)
        self._step(start, "depth", (), self.root_position)

    def move(self, uci):
        self._require_thought("move_after_final")
        if not self.visits:
            raise _Illegal("move_before_depth")
        visit = self.visits[-1]
        if visit.concluded:
            raise _Illegal("move_after_conclusion")
        if self.tracker.search_terminal() is not None:
            raise _Illegal("move_after_terminal")
        try:
            parsed = _board_move(uci)
        except ValueError as error:
            raise _Illegal("illegal_move", str(error)) from error
        if parsed not in self.tracker.board.legal_moves:
            raise _Illegal("illegal_move", f"not legal at this node: {uci}")
        if not self.tracker.path:
            self.round_has_candidate = True
        visit.explored = True
        visit.written = True
        start = len(self.tokens)
        san = self.tracker.board.san(parsed)
        token = self._move_token(uci)
        self.tracker.push(uci)
        self.visits.append(_Visit(self.tracker.path))
        self._emit(token, ROLE_MOVE)
        self._step(
            start,
            "move",
            self.tracker.path,
            self.tracker.fen(),
            uci=uci,
            san=san,
        )

    def _leave(self):
        child = self.tracker.path[-1]
        self.visits.pop()
        parent = self.visits[-1]
        if child not in parent.exact and child not in parent.refuted_children:
            parent.passed_children.add(child)

    def back(self):
        self._require_thought("back_after_final")
        if not self.tracker.path:
            raise _Illegal("back_at_root")
        start = len(self.tokens)
        self._leave()
        self.tracker.back()
        self._emit(BACK, ROLE_CONTROL)
        self._step(start, "back", self.tracker.path, self.tracker.fen())

    def reset(self):
        self._require_thought("reset_after_final")
        if not self.visits:
            raise _Illegal("reset_before_depth")
        if not self.tracker.path and self.visits[0].evaluated:
            raise _Illegal("reset_after_root_evaluation")
        start = len(self.tokens)
        while self.tracker.path:
            self._leave()
            self.tracker.back()
        self._emit(RESET, ROLE_CONTROL)
        self._step(start, "reset", self.tracker.path, self.tracker.fen())

    def evaluate_token(self, token):
        self._require_thought("evaluation_after_final")
        if not is_eval_token(token):
            raise _Illegal("unexpected_token", f"token {token}")
        if not self.visits:
            raise _Illegal("evaluation_before_depth")
        visit = self.visits[-1]
        if visit.concluded:
            raise _Illegal("evaluation_after_conclusion")
        start = len(self.tokens)
        self._emit(token, ROLE_EVAL)
        visit.evaluated = True
        visit.written = True
        if self.tracker.path:
            self.visits[-2].exact[self.tracker.path[-1]] = token
        distance, win = _mate_fields(token)
        self._step(
            start,
            "evaluate",
            self.tracker.path,
            self.tracker.fen(),
            evaluation=decode_value_token(token),
            mate=distance,
            mate_win=win,
        )

    def refuted(self):
        self._require_thought("refuted_after_final")
        if not self.visits or not self.tracker.path:
            raise _Illegal("refuted_at_root")
        visit = self.visits[-1]
        if visit.written:
            raise _Illegal("refuted_after_content")
        if not self._reference_exists():
            raise _Illegal("refuted_without_reference")
        start = len(self.tokens)
        child = self.tracker.path[-1]
        parent = self.visits[-2]
        self._emit(REFUTED, ROLE_REFUTED)
        visit.refuted = True
        visit.written = True
        parent.refuted_children.add(child)
        parent.passed_children.discard(child)
        self._step(start, "refuted", self.tracker.path, self.tracker.fen())

    def _expected_board(self):
        observation = self.tracker.board_tokens()
        if self.mirrored:
            observation = TOKEN_MIRROR[observation]
        return [int(token) for token in observation]

    def _write_board(self, start, observation, emit):
        try:
            observed, observed_repetition = _decode_observation(
                observation, self.mirrored
            )
        except ValueError as error:
            raise _Illegal("board_invalid", str(error)) from error
        observed_fields = observed.split()
        true_fields = self.tracker.fen().split()
        repetition = self.tracker.repetition()
        if (
            observed_fields[:4] != true_fields[:4]
            or observed_fields[4] != str(min(int(true_fields[4]), 100))
            or observed_repetition != min(repetition, 5)
        ):
            raise _Illegal("board_mismatch", observed)
        if emit:
            self._emit(BOARD, ROLE_CONTROL)
            for token in observation:
                self._emit(int(token), ROLE_BOARD)
        self.visits[-1].written = True
        self._step(
            start,
            "board",
            self.tracker.path,
            self.tracker.fen(),
            observed=observed,
            repetition=repetition,
        )

    def board(self, observation=None):
        self._require_thought("board_after_final")
        if not self.visits or not self.tracker.path:
            raise _Illegal("board_at_root")
        if observation is None:
            observation = self._expected_board()
        else:
            observation = [int(token) for token in observation]
        self._write_board(len(self.tokens), observation, True)

    def _begin_board(self):
        self._require_thought("board_after_final")
        if not self.visits or not self.tracker.path:
            raise _Illegal("board_at_root")
        self.pending_board = (len(self.tokens), self._expected_board(), 0)
        self._emit(BOARD, ROLE_CONTROL)

    def _continue_board(self, token):
        start, observation, index = self.pending_board
        if token != observation[index]:
            raise _Illegal("board_mismatch", f"token {token}")
        self._emit(token, ROLE_BOARD)
        index += 1
        if index == len(observation):
            self.pending_board = None
            self._write_board(start, observation, False)
        else:
            self.pending_board = (start, observation, index)

    # --- final decision and label suffix --- #

    def _decision_fields(self, uci):
        if uci == CLAIM_DRAW_UCI:
            if not self._claimable():
                raise _Illegal("final_illegal", "draw is not claimable at the root")
            return CLAIM_DRAW, None, True
        try:
            move = _board_move(uci)
        except ValueError as error:
            raise _Illegal("final_illegal", str(error)) from error
        if move not in self.root.legal_moves:
            raise _Illegal("final_illegal", f"not a legal root move: {uci}")
        if self.root_terminal is not None:
            raise _Illegal("final_illegal", "the root is a rule terminal")
        return self._move_token(uci), self.root.san(move), False

    def begin_final(self):
        self._require_thought("final_after_final")
        if not self.decision_tokens:
            raise _Illegal("final_illegal", "the root has no legal decision")
        self._final_start = len(self.tokens)
        self._final_path = self.tracker.path
        self._final_fen = self.tracker.fen()
        self._emit(FINAL, ROLE_CONTROL)
        self.suffix = _DECISION

    def final_decision(self, uci):
        if self.suffix != _DECISION:
            raise _Illegal("unexpected_token", "not at the final decision")
        token, san, claim = self._decision_fields(uci)
        self._emit(token, ROLE_FINAL_MOVE)
        self._step(
            self._final_start,
            "final",
            self._final_path,
            self._final_fen,
            uci=uci,
            san=san,
        )
        self.final_move = uci
        self.suffix = _VALUE
        self.claim = claim
        if not claim:
            self.pv_tracker = PathTracker(self.root_fen, self.tracker.root_repetition)
            self.pv_tracker.push(uci)

    def final(self, uci):
        """Write FINAL and its decision in one call."""
        self._decision_fields(uci)
        self.begin_final()
        self.final_decision(uci)

    def final_value_token(self, token):
        if self.suffix != _VALUE or self.value_written:
            raise _Illegal("final_without_value")
        if not VALUE <= token < MATE_WIN:
            raise _Illegal("final_without_value", f"token {token}")
        start = len(self.tokens)
        self._emit(token, ROLE_FINAL_EVAL)
        self.value_written = True
        self.value_bin = token - VALUE
        self._step(
            start,
            "final_evaluate",
            (),
            self.root_position,
            evaluation=decode_value_token(token),
        )

    def final_mate_token(self, token):
        if not self.value_written:
            raise _Illegal("final_without_value")
        if self.suffix != _VALUE:
            raise _Illegal("mate_after_line")
        if self.mate_written:
            raise _Illegal("mate_repeated")
        if self.value_bin == VALUE_BINS - 1:
            if not MATE_WIN <= token < MATE_LOSS:
                raise _Illegal("mate_after_non_extreme", f"token {token}")
            distance, win = token - MATE_WIN, True
        elif self.value_bin == 0:
            if not MATE_LOSS <= token < PAD:
                raise _Illegal("mate_after_non_extreme", f"token {token}")
            distance, win = token - MATE_LOSS, False
        else:
            raise _Illegal("mate_after_non_extreme", f"value bin {self.value_bin}")
        start = len(self.tokens)
        self._emit(token, ROLE_FINAL_EVAL)
        self.mate_written = True
        self.suffix = _MATE
        self._step(
            start,
            "final_evaluate",
            (),
            self.root_position,
            evaluation=decode_value_token(token),
            mate=distance,
            mate_win=win,
        )

    def final_pv(self, uci):
        if self.phase == _DONE:
            raise _Illegal("trailing_tokens")
        if not self.value_written:
            raise _Illegal("final_without_value")
        if self.claim:
            raise _Illegal("claim_with_variation")
        if self.pv_tracker.search_terminal() is not None:
            raise _Illegal("final_pv_after_terminal")
        try:
            parsed = _board_move(uci)
        except ValueError as error:
            raise _Illegal("final_illegal", str(error)) from error
        if parsed not in self.pv_tracker.board.legal_moves:
            raise _Illegal("final_illegal", f"not legal in the final line: {uci}")
        start = len(self.tokens)
        san = self.pv_tracker.board.san(parsed)
        token = self._move_token(uci)
        self.pv_tracker.push(uci)
        self._emit(token, ROLE_FINAL_PV)
        self.suffix = _PV
        self._step(
            start,
            "final_pv",
            self.pv_tracker.path,
            self.pv_tracker.fen(),
            uci=uci,
            san=san,
        )

    def end(self):
        if self.phase == _DONE:
            raise _Illegal("trailing_tokens")
        if not self.value_written:
            raise _Illegal("final_without_value")
        start = len(self.tokens)
        self._emit(END, ROLE_CONTROL)
        self.suffix = _DONE
        self.phase = _DONE
        if self.pv_tracker is not None:
            path, position = self.pv_tracker.path, self.pv_tracker.fen()
        else:
            path, position = (), self.root_position
        self._step(start, "end", path, position)

    def apply(self, token):
        """Consume one already-chosen token under the current state."""
        token = int(token)
        if self.phase == _DONE:
            raise _Illegal("trailing_tokens")
        if self.pending_board is not None:
            self._continue_board(token)
            return
        if self.suffix == _DECISION:
            if is_move_token(token):
                self.final_decision(_absolute_uci(token, self.mirrored))
                return
            raise _Illegal("final_without_move", f"token {token}")
        if self.suffix is not None:
            if token == END:
                self.end()
                return
            if not self.value_written:
                if VALUE <= token < MATE_WIN:
                    self.final_value_token(token)
                    return
                raise _Illegal("final_without_value", f"token {token}")
            if MATE_WIN <= token < PAD:
                self.final_mate_token(token)
                return
            if is_move_token(token):
                self.final_pv(_absolute_uci(token, self.mirrored))
                return
            raise _Illegal("unexpected_token", f"token {token}")
        if token == DEPTH:
            self.depth()
        elif token == REFUTED:
            self.refuted()
        elif token == BACK:
            self.back()
        elif token == RESET:
            self.reset()
        elif token == BOARD:
            self._begin_board()
        elif token == FINAL:
            self.begin_final()
        elif token == END:
            raise _Illegal("end_before_final")
        elif VALUE <= token < PAD:
            self.evaluate_token(token)
        elif is_move_token(token):
            self.move(_absolute_uci(token, self.mirrored))
        else:
            raise _Illegal("unexpected_token", f"token {token}")


class TraceWriter:
    """Accumulate one rule-consistent trace over a root position.

    Every operation advances the shared state machine, so the writer never
    stores a state the replay could not rebuild. ``final`` writes the decision
    point; its label suffix continues with ``final_evaluate``, ``final_pv``,
    and ``end``. ``apply`` consumes one chosen token, which is how free
    generation writes through the same state machine as replay.
    """

    def __init__(self, root_fen: str, repetition: int = 1, teacher_width: int = 1):
        self._state = _TraceState(root_fen, repetition, teacher_width)

    def copy(self):
        """Fork the grammar for a prospective composite-token expansion."""
        from copy import deepcopy

        result = object.__new__(TraceWriter)
        # Recorded steps are immutable; only the live grammar needs deep copying.
        result._state = deepcopy(
            self._state,
            {
                id(values): values.copy()
                for values in (self._state.tokens, self._state.roles, self._state.steps)
            },
        )
        return result

    @property
    def finished(self):
        return self._state.phase == _DONE

    @property
    def final_written(self):
        return self._state.suffix is not None

    @property
    def completing_board(self):
        return self._state.pending_board is not None

    @property
    def awaiting_final_move(self):
        return self._state.suffix == _DECISION

    @property
    def final_move(self):
        """The decision written after FINAL, or None before it is written."""
        return self._state.final_move

    @property
    def decision_tokens(self):
        """Root decision tokens: legal moves plus the claim when claimable."""
        return self._state.decision_tokens

    @property
    def mirrored(self):
        return self._state.mirrored

    @property
    def path(self):
        """Absolute UCI path from the root to the current thought node."""
        return self._state.tracker.path

    @property
    def tokens(self):
        return np.asarray(self._state.tokens, dtype=np.int16)

    @property
    def roles(self):
        return np.asarray(self._state.roles, dtype=np.int8)

    @property
    def steps(self):
        return tuple(self._state.steps)

    def legal_moves(self):
        """Absolute UCI moves that are legal at the current thought node."""
        return self._state.tracker.legal_moves()

    def allowed_tokens(self, final_allowed=False, cache=None):
        """Token ids the current state allows; the generation mask's source."""
        return self._state.allowed_tokens(final_allowed, cache)

    def apply(self, token):
        """Consume one chosen token, validating it against the state machine."""
        self._state.apply(token)

    def depth(self):
        """Open the next round at the root."""
        self._state.depth()

    def move(self, uci):
        """Enter the child of the current node reached by a legal board move."""
        self._state.move(uci)

    def back(self):
        """Return to the parent; illegal at the root."""
        self._state.back()

    def reset(self):
        """Return to the root from any depth."""
        self._state.reset()

    def refuted(self):
        """Declare that the just-entered candidate did not beat the reference."""
        self._state.refuted()

    def evaluate(self, value=None, mate=None, win=None):
        """Write an evaluation of the current node from the root player's view.

        ``mate`` is a nonnegative distance and ``win`` says whether the root
        player delivers it; the explicit flag keeps distance zero unambiguous.
        """
        if (value is None) == (mate is None):
            raise ValueError("Give exactly one of value or mate")
        if mate is not None:
            if not isinstance(win, bool):
                raise ValueError("A mate evaluation needs win=True or win=False")
            token = mate_token(mate, win)
        else:
            if win is not None:
                raise ValueError("win applies only to mate evaluations")
            token = VALUE + value_bin(value)
        self._state.evaluate_token(token)

    def board(self):
        """Write BOARD and the current position's encoded board tokens."""
        self._state.board()

    def final(self, uci):
        """Write FINAL and its root decision, which may be the draw claim."""
        self._state.final(uci)

    def begin_final(self):
        """Write FINAL alone; the decision follows in a separate model step."""
        self._state.begin_final()

    def final_decision(self, uci):
        """Write the decision after a separately written FINAL."""
        self._state.final_decision(uci)

    def final_evaluate(self, value=None, mate=None, win=None):
        """Write the label evaluation: a value token always, a mate on demand."""
        if (value is None) == (mate is None):
            raise ValueError("Give exactly one of value or mate")
        if mate is not None:
            if not isinstance(win, bool):
                raise ValueError("A mate evaluation needs win=True or win=False")
            token = mate_token(mate, win)
            self._state.final_value_token(VALUE + (VALUE_BINS - 1 if win else 0))
            self._state.final_mate_token(token)
        else:
            if win is not None:
                raise ValueError("win applies only to mate evaluations")
            self._state.final_value_token(VALUE + value_bin(value))

    def final_pv(self, uci):
        """Write one move of the best line that follows the FINAL move."""
        self._state.final_pv(uci)

    def end(self):
        """Close the trace with <end>."""
        self._state.end()


def replay(tokens, root_fen, repetition=1, teacher_width=1):
    """Rebuild and validate a trace from its tokens and the root position.

    The state is reconstructed with python-chess from the root FEN alone;
    writer intermediates are never consulted. Raises TraceError at the first
    broken rule, with a machine-readable tag and token index.
    """
    values = [int(token) for token in tokens]
    while values and values[-1] == PAD:
        values.pop()
    if PAD in values:
        raise TraceError("pad_inside_trace", values.index(PAD))
    state = _TraceState(root_fen, repetition, teacher_width)
    index = 0
    while index < len(values):
        token = values[index]
        try:
            if token == BOARD:
                observation, end = _read_observation(values, index)
                state.board(observation)
                index = end
            else:
                state.apply(token)
                index += 1
        except _Illegal as error:
            raise TraceError(error.tag, index, error.detail) from error
        if state.phase == _DONE:
            if index != len(values):
                raise TraceError("trailing_tokens", index)
            break
    if state.phase != _DONE:
        if state.suffix is None:
            raise TraceError("missing_final", len(values))
        if state.suffix == _DECISION:
            raise TraceError("final_without_move", len(values))
        if not state.value_written:
            raise TraceError("final_without_value", len(values))
        raise TraceError("missing_end", len(values))
    return TraceProgram(
        root_fen,
        repetition,
        tuple(state.tokens),
        tuple(state.roles),
        tuple(state.steps),
        state.final_move,
        state.mirrored,
    )


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
    """A schema-3 or schema-4 record, indexed by event type."""

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

    @cached_property
    def _candidates_by_attempt(self):
        result = {}
        for event in self.candidates:
            result.setdefault((event["round"], event["attempt"]), []).append(event)
        return result

    @cached_property
    def _candidate_groups_cache(self):
        return {}

    @cached_property
    def _frontiers_by_path(self):
        return {
            (event["round"], event["attempt"], tuple(event["path"])): event
            for event in self.frontiers
        }

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


def decode_sidecar(lines):
    """Decode output from the qualified producer without replaying its search.

    ``parse_sidecar`` supplies the exhaustive diagnostic checks used when
    qualifying the producer or investigating an invalid record. Any malformed
    input surfaces as ``SynthesisError``; structural accidents never escape
    as bare ``KeyError``/``IndexError``/``TypeError``.
    """
    try:
        events = tuple(orjson.loads(line) for line in lines)
        if not events or events[-1]["type"] != "search_end":
            raise SynthesisError("the engine sidecar did not finish")
        by_type = {kind: [] for kind in EVENT_TYPES}
        for event in events:
            by_type[event["type"]].append(event)
        return Sidecar(
            events,
            by_type["session"][0],
            by_type["position"][0],
            by_type["search_start"][0],
            tuple(by_type["round_start"]),
            tuple(by_type["attempt"]),
            tuple(by_type["candidate"]),
            tuple(by_type["frontier"]),
            tuple(by_type["round_end"]),
            events[-1],
        )
    except (ValueError, KeyError, IndexError, TypeError) as error:
        if isinstance(error, SynthesisError):
            raise
        raise SynthesisError(f"malformed sidecar: {error}") from error


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


PRESENTATIONS = ("zero", "full", "random")


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


class Boundary(NamedTuple):
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
    trace_text: str


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
    __slots__ = ("move", "parent", "order", "exact", "refuted", "first_seq")

    def __init__(self, move, parent, order, calls):
        self.move = move
        self.parent = parent
        self.order = order
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


class _CandidateGroups:
    """Group raw calls once; build candidates only at narrated parents."""

    def __init__(self, events):
        self.calls = {}
        self.cache = {}
        for event in events:
            parent = tuple(event["path"][:-1])
            moves = self.calls.setdefault(parent, {})
            moves.setdefault(event["move"], []).append(event)

    def get(self, parent, default=()):
        if parent in self.cache:
            return self.cache[parent]
        calls_by_move = self.calls.get(parent)
        if calls_by_move is None:
            return default
        candidates = [
            _Candidate(move, parent, calls[0]["order"], calls)
            for move, calls in calls_by_move.items()
        ]
        candidates.sort(key=lambda candidate: (candidate.order, candidate.first_seq))
        self.cache[parent] = candidates
        return candidates


def _candidate_groups(sidecar, round_number, attempt):
    key = (round_number, attempt)
    if key not in sidecar._candidate_groups_cache:
        sidecar._candidate_groups_cache[key] = _CandidateGroups(
            sidecar._candidates_by_attempt.get(key, ())
        )
    return sidecar._candidate_groups_cache[key]


def _frontier_for(sidecar, round_number, attempt, path):
    return sidecar._frontiers_by_path.get((round_number, attempt, path))


class _RootState:
    """The champion visible at a truncation boundary."""

    def __init__(self, mirrored, teacher_width):
        self.mirrored = mirrored
        self.teacher_width = teacher_width
        self.round = 0
        self.current_round = {}
        self.round_trace = {}
        self.round_count = 0
        self.previous = None
        self.current = None

    def begin_round(self, round_number):
        self.round = round_number
        self.current_round = {}
        self.round_trace = {}
        self.round_count = 0
        self.current = self.previous

    def record_exact(self, move, raw, token):
        trace = self.round_trace.get(move)
        if trace is None:
            self.round_count += 1
            trace = self.round_count
            self.round_trace[move] = trace
        self.current_round[move] = (raw, token, trace)
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


class _TraceBuffer:
    """Encode known synthesis operations without running the grammar checker."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.mirrored = tracker.board.turn == chess.BLACK
        self.tokens = []
        self.roles = []
        self.last_was_board = False
        self.final_start = None
        self.final_move = None
        self.text_parts = []
        self.text_ends = []
        self.text_line = None
        self.round = 0
        fen, repetition = decode_board(tracker.board_tokens())
        self.initial_text = f"{fen} (repetition {repetition})"

    def _text(self, text):
        self.text_parts.append(text)
        self.text_ends.append(len(self.tokens))

    @staticmethod
    def _evaluation_text(token):
        if token >= MATE_WIN:
            distance, win = _mate_fields(token)
            return f"{'win' if win else 'loss'}:{distance}"
        return f"value:{decode_value_token(token):+.3f}"

    def _emit(self, token, role):
        self.tokens.append(token)
        self.roles.append(role)
        self.last_was_board = False

    def depth(self):
        self._emit(DEPTH, ROLE_CONTROL)
        self.round += 1
        self._text(f"\nDEPTH {self.round}")
        self.text_line = "depth"

    def move(self, uci, san):
        self._emit(_move_token(uci, self.mirrored), ROLE_MOVE)
        self._text((" " if self.text_line == "candidate" else "\n") + san)
        self.text_line = "candidate"

    def back(self):
        self._emit(BACK, ROLE_CONTROL)
        self._text(" BACK")

    def reset(self):
        self._emit(RESET, ROLE_CONTROL)
        self._text("\nRESET")
        self.text_line = "reset"

    def refuted(self):
        self._emit(REFUTED, ROLE_REFUTED)
        self._text(" REFUTED")

    def apply(self, token):
        self._emit(token, ROLE_EVAL)
        evaluation = self._evaluation_text(token)
        if self.tracker.depth:
            self._text(" " + evaluation)
        else:
            self._text("\nroot " + evaluation)
            self.text_line = "root"

    def board(self):
        observation = self.tracker.board_tokens()
        fen, _ = decode_board(observation)
        if self.mirrored:
            observation = TOKEN_MIRROR[observation]
        self._emit(BOARD, ROLE_CONTROL)
        self.tokens.extend(observation)
        self.roles.extend([ROLE_BOARD] * len(observation))
        self.last_was_board = True
        self._text(f" BOARD {fen} (repetition {self.tracker.repetition()})")

    def final(self, uci):
        self.final_start = len(self.tokens)
        self.final_move = uci
        self._emit(FINAL, ROLE_CONTROL)
        self._emit(
            CLAIM_DRAW if uci == CLAIM_DRAW_UCI else _move_token(uci, self.mirrored),
            ROLE_FINAL_MOVE,
        )
        self.pv_board = chess.Board(self.tracker.root_fen)
        san = (
            "CLAIM_DRAW"
            if uci == CLAIM_DRAW_UCI
            else self.pv_board.san_and_push(chess.Move.from_uci(uci))
        )
        self._text("\nFINAL " + san)

    def final_evaluate(self, value=None, mate=None, win=None):
        if mate is None:
            self._emit(VALUE + value_bin(value), ROLE_FINAL_EVAL)
            self._text(" " + self._evaluation_text(self.tokens[-1]))
        else:
            self._emit(VALUE + (VALUE_BINS - 1 if win else 0), ROLE_FINAL_EVAL)
            self._text(" " + self._evaluation_text(self.tokens[-1]))
            self._emit(mate_token(mate, win), ROLE_FINAL_EVAL)
            self._text(" " + self._evaluation_text(self.tokens[-1]))

    def final_pv(self, uci):
        self._emit(_move_token(uci, self.mirrored), ROLE_FINAL_PV)
        self._text(" " + self.pv_board.san_and_push(chess.Move.from_uci(uci)))

    def end(self):
        self._emit(END, ROLE_CONTROL)
        self._text(" END")


class _Synthesis:
    def __init__(self, sidecar, root, config, seed):
        self.sidecar = sidecar
        self.root = root
        self.config = config
        self.seed = config.seed if seed is None else seed
        self.tracker = PathTracker(sidecar.fen, int(root["repetition"]))
        self.writer = _TraceBuffer(self.tracker)
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
        if not self.writer.last_was_board:
            self.no_tool_len += delta
        best, value, basis = self.state.snapshot()
        self.boundaries.append(Boundary(length, self.no_tool_len, best, value, basis))
        self.previous_len = length

    def _write_move(self, move, spine=True):
        san = self.tracker.push_legal(move)
        self.writer.move(move, san)
        self._record()
        if not spine:
            return
        self.plies_since_board += 1
        if self.board_rng.random() < self.config.synthesis.board_descent_hazard:
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
                token = mate_token(0, len(child_path) % 2 == 1)
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
            if self.board_rng.random() < self.config.synthesis.board_frontier_p:
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
                    None if wdl is not None else _token_mate(token),
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
        suffix_start = self.writer.final_start
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
        suffix_text_start = bisect_right(self.writer.text_ends, suffix_start)
        suffix_text = "".join(self.writer.text_parts[suffix_text_start:])
        presentations = (
            (self.root["presentation"],)
            if "presentation" in self.root
            else PRESENTATIONS
        )
        for presentation in presentations:
            if presentation == "zero":
                cut = 0
                boundaries = ()
            elif presentation == "full":
                cut = suffix_start
                boundaries = tuple(self.boundaries)
            elif presentation == "random":
                random_rng = random.Random(f"{self.seed}:{self.root['root_id']}:random")
                low = math.ceil(
                    self.config.synthesis.random_min_fraction * suffix_start
                )
                high = math.floor(
                    self.config.synthesis.random_max_fraction * suffix_start
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
                self.writer.initial_text
                + "".join(
                    self.writer.text_parts[: bisect_right(self.writer.text_ends, cut)]
                )
                + suffix_text,
            )
        stats = self._stats()
        return SynthesizedRoot(self.root, rows, tuple(self.round_audits), stats)

    def _stats(self):
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
            "claim_draw": self._claim,
            "claimable": self.tracker.root_repetition >= 3
            or chess.Board(self.sidecar.fen).halfmove_clock >= 100,
            "thinking_tokens": self.writer.final_start,
            "total_tokens": len(self.writer.tokens),
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
    """Synthesize the assigned presentation, or all for an unassigned diagnostic root."""
    validate_root(root, sidecar)
    if sidecar.search_start["depth"] != sidecar.round_ends[-1]["depth"]:
        raise SynthesisError("the recorded search did not complete its requested depth")
    return _Synthesis(sidecar, root, config, seed).synthesize()
