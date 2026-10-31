"""Sequence trace programs over the unified chess vocabulary.

A trace is a causal program over one root position. Move tokens walk a path of
absolute UCI moves; BACK returns to the parent, RESET returns to the root from
any depth, value and mate tokens evaluate the current node from the root
player's perspective, BOARD asks for the full encoded observation of the
current node, and ``FINAL <root move>`` ends the program immediately. A later
evaluation of a node corrects an earlier one.

Stored moves are rank-reflected when the root has Black to move, matching the
board prefix's canonical frame. Controls, value tokens, and board responses
stay in that frame too: a response is the token mirror of the current absolute
position, so its side and counter metadata keep their absolute values while
pieces, castling rights, and the en-passant square are reflected. Counters
above the vocabulary's range share its top atom.

``TraceWriter`` is the only producer. ``replay`` rebuilds every state from the
token stream and the root FEN with python-chess and never reads writer state;
it returns the validated steps and their natural-language rendering.
"""

from dataclasses import dataclass

import chess
import numpy as np

from pondera.chess.codec import decode_board, decode_value_token, mate_token, value_bin
from pondera.chess.symmetry import TOKEN_MIRROR, reflect_uci
from pondera.chess.tracker import PathTracker
from pondera.chess.vocabulary import (
    BACK,
    BOARD,
    FINAL,
    IDX_TO_UCI_MOVE,
    MATE_LOSS,
    MATE_WIN,
    META_END,
    MOVE,
    PAD,
    RESET,
    SIDE,
    UCI_MOVE_TO_IDX,
    VALUE,
)

# The grammar version recorded in sequence dataset manifests.
TRACE_GRAMMAR_VERSION = 1

# Per-token roles: the loss sees generated moves, evaluations, and controls,
# treats a tool response as context only, and weights the final move apart.
ROLE_PAD = 0
ROLE_TOOL = 1
ROLE_MOVE = 2
ROLE_EVAL = 3
ROLE_CONTROL = 4
ROLE_FINAL_MOVE = 5
ROLE_COUNT = 6

# One board observation carries eight metadata atoms after its piece atoms.
OBSERVATION_METADATA = 8


class TraceError(ValueError):
    """A malformed or rule-inconsistent trace, tagged at its first bad token."""

    def __init__(self, tag, index, detail=""):
        self.tag = tag
        self.index = index
        message = f"{tag} at token {index}"
        if detail:
            message += f": {detail}"
        super().__init__(message)


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
    token's decoded scalar.
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
        return len(self.tokens) - 1

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
                lines.append(
                    f"{number}. {span} FINAL {step.san} ({step.uci}) from the root; "
                    f"thought ended at path {where}; position {step.position}"
                )
        return "\n".join(lines)


class TraceWriter:
    """Accumulate one rule-consistent trace over a root position.

    Every operation advances the tracker, so the writer never stores a state
    the replay could not rebuild. Writing after FINAL raises; the final move is
    checked against the root's legal set, not the current node's.
    """

    def __init__(self, root_fen: str, repetition: int = 1):
        self._tracker = PathTracker(root_fen, repetition)
        self._root = self._tracker.board.copy()
        self._mirror = self._root.turn == chess.BLACK
        self._tokens = []
        self._roles = []
        self._steps = []
        self._finished = False

    def _require_open(self):
        if self._finished:
            raise ValueError("Trace already ended with FINAL")

    def _move_token(self, uci):
        framed = reflect_uci(uci) if self._mirror else uci
        return MOVE + UCI_MOVE_TO_IDX[framed]

    def _append(self, token, role):
        self._tokens.append(int(token))
        self._roles.append(role)

    def _step(self, start, operation, **fields):
        self._steps.append(
            Step(
                start,
                len(self._tokens),
                operation,
                self._tracker.path,
                self._tracker.fen(),
                **fields,
            )
        )

    def move(self, uci):
        """Enter the child of the current node reached by a legal board move."""
        self._require_open()
        start = len(self._tokens)
        san = self._tracker.board.san(_board_move(uci))
        self._tracker.push(uci)
        self._append(self._move_token(uci), ROLE_MOVE)
        self._step(start, "move", uci=uci, san=san)

    def back(self):
        """Return to the parent; illegal at the root."""
        self._require_open()
        start = len(self._tokens)
        self._tracker.back()
        self._append(BACK, ROLE_CONTROL)
        self._step(start, "back")

    def reset(self):
        """Return to the root from any depth."""
        self._require_open()
        start = len(self._tokens)
        self._tracker.reset()
        self._append(RESET, ROLE_CONTROL)
        self._step(start, "reset")

    def evaluate(self, value=None, mate=None, win=None):
        """Write an evaluation of the current node from the root player's view.

        ``mate`` is a nonnegative distance and ``win`` says whether the root
        player delivers it; the explicit flag keeps distance zero unambiguous.
        """
        self._require_open()
        if (value is None) == (mate is None):
            raise ValueError("Give exactly one of value or mate")
        if mate is not None:
            if not isinstance(win, bool):
                raise ValueError("A mate evaluation needs win=True or win=False")
            token = mate_token(mate, win)
            distance, win = mate, win
        else:
            if win is not None:
                raise ValueError("win applies only to mate evaluations")
            token = VALUE + value_bin(value)
            distance, win = None, None
        start = len(self._tokens)
        self._append(token, ROLE_EVAL)
        self._step(
            start,
            "evaluate",
            evaluation=decode_value_token(token),
            mate=distance,
            mate_win=win,
        )

    def board(self):
        """Request and write the full observation of the current node.

        The request carries loss; the appended response is external context.
        A black-to-move root mirrors the response while leaving its side and
        counter atoms absolute.
        """
        self._require_open()
        start = len(self._tokens)
        observation = self._tracker.board_tokens()
        if self._mirror:
            observation = TOKEN_MIRROR[observation]
        self._append(BOARD, ROLE_CONTROL)
        for token in observation:
            self._append(int(token), ROLE_TOOL)
        self._step(
            start,
            "board",
            observed=_decode_observation(observation, self._mirror)[0],
            repetition=self._tracker.repetition(),
        )

    def final(self, uci):
        """Write the root decision and end the trace immediately."""
        self._require_open()
        start = len(self._tokens)
        move = _board_move(uci)
        if move not in self._root.legal_moves:
            raise ValueError(f"FINAL needs a legal root move: {uci}")
        san = self._root.san(move)
        self._append(FINAL, ROLE_CONTROL)
        self._append(self._move_token(uci), ROLE_FINAL_MOVE)
        self._step(start, "final", uci=uci, san=san)
        self._finished = True

    @property
    def finished(self):
        return self._finished

    @property
    def mirrored(self):
        return self._mirror

    @property
    def path(self):
        """Absolute UCI path from the root to the current node."""
        return self._tracker.path

    def legal_moves(self):
        """Absolute UCI moves that are legal at the current node."""
        return self._tracker.legal_moves()

    @property
    def tokens(self):
        return np.asarray(self._tokens, dtype=np.int16)

    @property
    def roles(self):
        return np.asarray(self._roles, dtype=np.int8)

    @property
    def steps(self):
        return tuple(self._steps)


def _board_move(uci):
    """Parse an absolute UCI board move, rejecting the draw-claim pseudo-move."""
    try:
        return chess.Move.from_uci(uci)
    except ValueError as error:
        raise ValueError(f"Not a board move: {uci}") from error


def _absolute_uci(token, mirrored):
    uci = IDX_TO_UCI_MOVE[token - MOVE]
    return reflect_uci(uci) if mirrored else uci


def _root_san(root, uci, index):
    try:
        move = _board_move(uci)
    except ValueError as error:
        raise TraceError("final_illegal", index, str(error)) from error
    if move not in root.legal_moves:
        raise TraceError("final_illegal", index, f"not legal at the root: {uci}")
    return root.san(move)


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


def replay(tokens, root_fen, repetition=1):
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
    tracker = PathTracker(root_fen, repetition)
    root = tracker.board.copy()
    mirrored = root.turn == chess.BLACK
    roles, steps = [], []
    index, final_move = 0, None
    while index < len(values):
        token = values[index]
        start = index
        if is_move_token(token):
            uci = _absolute_uci(token, mirrored)
            try:
                san = tracker.board.san(_board_move(uci))
                tracker.push(uci)
            except ValueError as error:
                raise TraceError("illegal_move", index, str(error)) from error
            roles.append(ROLE_MOVE)
            steps.append(
                Step(start, start + 1, "move", tracker.path, tracker.fen(), uci, san)
            )
            index += 1
        elif token == BACK:
            if not tracker.path:
                raise TraceError("back_at_root", index)
            tracker.back()
            roles.append(ROLE_CONTROL)
            steps.append(Step(start, start + 1, "back", tracker.path, tracker.fen()))
            index += 1
        elif token == RESET:
            tracker.reset()
            roles.append(ROLE_CONTROL)
            steps.append(Step(start, start + 1, "reset", tracker.path, tracker.fen()))
            index += 1
        elif is_eval_token(token):
            distance, win = _mate_fields(token)
            roles.append(ROLE_EVAL)
            steps.append(
                Step(
                    start,
                    start + 1,
                    "evaluate",
                    tracker.path,
                    tracker.fen(),
                    evaluation=decode_value_token(token),
                    mate=distance,
                    mate_win=win,
                )
            )
            index += 1
        elif token == BOARD:
            observation, end = _read_observation(values, index)
            try:
                observed, observed_repetition = _decode_observation(
                    observation, mirrored
                )
            except ValueError as error:
                raise TraceError("board_invalid", index, str(error)) from error
            observed_fields = observed.split()
            true_fields = tracker.fen().split()
            true_repetition = tracker.repetition()
            if (
                observed_fields[:4] != true_fields[:4]
                or observed_fields[4] != str(min(int(true_fields[4]), 100))
                or observed_repetition != min(true_repetition, 5)
            ):
                raise TraceError("board_mismatch", index, observed)
            roles.append(ROLE_CONTROL)
            roles.extend([ROLE_TOOL] * (end - start - 1))
            steps.append(
                Step(
                    start,
                    end,
                    "board",
                    tracker.path,
                    tracker.fen(),
                    observed=observed,
                    repetition=true_repetition,
                )
            )
            index = end
        elif token == FINAL:
            if index + 1 >= len(values) or not is_move_token(values[index + 1]):
                raise TraceError("final_without_move", index)
            uci = _absolute_uci(values[index + 1], mirrored)
            san = _root_san(root, uci, index)
            roles.extend([ROLE_CONTROL, ROLE_FINAL_MOVE])
            steps.append(
                Step(start, start + 2, "final", tracker.path, tracker.fen(), uci, san)
            )
            final_move = uci
            index += 2
            if index != len(values):
                raise TraceError("trailing_tokens", index)
            break
        else:
            raise TraceError("unexpected_token", index, f"token {token}")
    if final_move is None:
        raise TraceError("missing_final", len(values))
    return TraceProgram(
        root_fen,
        repetition,
        tuple(values),
        tuple(roles),
        tuple(steps),
        final_move,
        mirrored,
    )
