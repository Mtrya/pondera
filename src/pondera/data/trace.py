"""Sequence trace programs over the unified chess vocabulary.

A trace is a causal program over one root position. Move tokens walk a path of
absolute UCI moves; BACK returns to the parent, RESET returns to the root from
any depth, and value and mate tokens evaluate the current node from the root
player's perspective. ``BOARD`` asks for the full encoded observation of the
current node. ``<depth>`` opens a round at the root, and ``<refuted>`` states
that a just-entered candidate did not beat the reference at its parent.

Grammar version 2 groups the thought into rounds. A round is one root visit:
``<depth>``, the spine and candidate scan below it, and at most one round-end
root evaluation. Every non-root node has one visit from the move that enters
it to the BACK or RESET that leaves it; a visit records an exact outcome (one
evaluation), a refutation, or a pass, and writing a conclusion ends it.

The decision point is ``FINAL``, followed by a uniform label suffix:
``FINAL <move> <value> [<mate>] <best line...> <end>``. A root that is
claimable as a draw and whose label is a draw claim writes
``FINAL <claim_draw> <value> [<mate>] <end>`` instead. The label suffix is a
prediction region: it always comes from the deepest round of the complete
trace, so it is not audited against the visible thought when the trace was
truncated.

Stored moves are rank-reflected when the root has Black to move, matching the
board prefix's canonical frame. Controls, value tokens, and board responses
stay in that frame too: a response is the token mirror of the current absolute
position, so its side and counter metadata keep their absolute values while
pieces, castling rights, and the en-passant square are reflected. Counters
above the vocabulary's range share its top atom.

``TraceWriter`` is the only producer. ``replay`` rebuilds every state from the
token stream and the root FEN with python-chess and never reads writer state;
the writer's legality checks and ``replay`` share one state machine, and the
generation mask reads its allowed set, so the three sides cannot drift.
"""

from dataclasses import dataclass

import chess
import numpy as np

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

# The grammar version recorded in sequence dataset manifests.
TRACE_GRAMMAR_VERSION = 2

# Per-token roles: the loss sees every generated token, treats a tool response
# as context only, and separates the FINAL label into its decision, evaluation,
# and best line. A refutation states a candidate's outcome without a value.
ROLE_PAD = 0
ROLE_TOOL = 1
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

    def board(self, observation=None):
        self._require_thought("board_after_final")
        if not self.visits or not self.tracker.path:
            raise _Illegal("board_at_root")
        visit = self.visits[-1]
        start = len(self.tokens)
        true_fen = self.tracker.fen()
        if observation is None:
            observation = self.tracker.board_tokens()
            if self.mirrored:
                observation = TOKEN_MIRROR[observation]
            observed = _decode_observation(observation, self.mirrored)[0]
            repetition = self.tracker.repetition()
        else:
            observation = [int(token) for token in observation]
            try:
                observed, observed_repetition = _decode_observation(
                    observation, self.mirrored
                )
            except ValueError as error:
                raise _Illegal("board_invalid", str(error)) from error
            observed_fields = observed.split()
            true_fields = true_fen.split()
            repetition = self.tracker.repetition()
            if (
                observed_fields[:4] != true_fields[:4]
                or observed_fields[4] != str(min(int(true_fields[4]), 100))
                or observed_repetition != min(repetition, 5)
            ):
                raise _Illegal("board_mismatch", observed)
        self._emit(BOARD, ROLE_CONTROL)
        for token in observation:
            self._emit(int(token), ROLE_TOOL)
        visit.written = True
        self._step(
            start,
            "board",
            self.tracker.path,
            self.tracker.fen(),
            observed=observed,
            repetition=repetition,
        )

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
            raise _Illegal("unexpected_token", "BOARD needs a response")
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

    @property
    def finished(self):
        return self._state.phase == _DONE

    @property
    def final_written(self):
        return self._state.suffix is not None

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
        """Request and write the full observation of the current node.

        The request carries loss; the appended response is external context.
        A black-to-move root mirrors the response while leaving its side and
        counter atoms absolute.
        """
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
