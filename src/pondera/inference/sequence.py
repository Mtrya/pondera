"""Legal-masked free generation over the sequence trace grammar.

A thought is the trace program of ``pondera.data.trace``: moves walk the path,
BACK returns to the parent, RESET returns to the root, value and mate tokens
evaluate the current node, and ``FINAL <root move>`` ends the program. ``BOARD``
is disabled here, so every generated token is either navigation, an
evaluation, or the final decision. The generator writes through
``TraceWriter`` and validates the result with ``replay``, which keeps
black-to-move roots in the same canonical rank-reflected frame as training
data and rejects any token the rule state does not allow.

Two protocols bound a thought. A fixed budget masks FINAL until the requested
number of thinking tokens has been generated, then forces FINAL and one root
move; a budget of zero is a direct answer. Autonomous generation allows FINAL
at every boundary and stops when the model emits it, or at a declared safety
cap. Thinking tokens are every generated token before FINAL: moves, BACK,
RESET, and evaluations. The FINAL delimiter and its root move never count
toward the budget, and a BOARD response could not count because BOARD is off.

The final decision's distribution is the model's post-FINAL distribution
restricted to the legal root moves. Keeping it on the generated trace lets
callers score any legal target move without another forward pass.
"""

import math
from dataclasses import dataclass

import chess
import numpy as np
import torch

from pondera.chess.codec import decode_value_token, encode_boards
from pondera.chess.symmetry import reflect_uci
from pondera.chess.vocabulary import (
    BACK,
    BOARD,
    CHESS_VOCAB_SIZE,
    FINAL,
    IDX_TO_UCI_MOVE,
    MATE_LOSS,
    MATE_WIN,
    MOVE,
    PAD,
    RESET,
    SIDE,
    UCI_MOVE_TO_IDX,
    VALUE,
)
from pondera.data.trace import OBSERVATION_METADATA, TraceError, TraceWriter, replay

FIXED = "fixed"
AUTONOMOUS = "autonomous"

# Why generation stopped: the model emitted FINAL, the fixed budget forced it,
# or the autonomous safety cap forced it.
STOP_FINAL = "final"
STOP_BUDGET = "budget"
STOP_CAP = "cap"

# Fixed-budget points as fractions of the trace-length bound.
BUDGET_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)

_THOUGHT = "thought"
_ROOT_MOVE = "root_move"
_DONE = "done"


@dataclass(frozen=True)
class GenerationProtocol:
    """One fixed-budget or autonomous stopping rule."""

    kind: str
    budget: int | None = None
    safety_cap: int | None = None

    def __post_init__(self):
        if self.kind == FIXED:
            if self.budget is None or self.budget < 0 or self.safety_cap is not None:
                raise ValueError("A fixed protocol needs a nonnegative budget")
        elif self.kind == AUTONOMOUS:
            if (
                self.safety_cap is None
                or self.safety_cap < 0
                or self.budget is not None
            ):
                raise ValueError("An autonomous protocol needs a nonnegative cap")
        else:
            raise ValueError(f"Unknown generation protocol: {self.kind}")

    @property
    def limit(self):
        """The thinking-token count at which FINAL is forced."""
        return self.budget if self.kind == FIXED else self.safety_cap


def fixed_budget(budget: int) -> GenerationProtocol:
    """Mask FINAL before ``budget`` thinking tokens, then force it."""
    return GenerationProtocol(FIXED, budget=budget)


def autonomous(safety_cap: int) -> GenerationProtocol:
    """Allow FINAL anywhere, forcing it only at the declared safety cap."""
    return GenerationProtocol(AUTONOMOUS, safety_cap=safety_cap)


@dataclass(frozen=True)
class RootPosition:
    """One frozen root position and its repetition count."""

    fen: str
    repetition: int = 1

    def __post_init__(self):
        if self.repetition < 1:
            raise ValueError("Root repetition must be positive")


@dataclass(frozen=True)
class GeneratedTrace:
    """One generated trace and the decision distribution that closed it.

    ``tokens`` is the whole model-generated trace including FINAL and its root
    move. ``thinking_tokens`` counts only the tokens before FINAL.
    ``final_move`` is an absolute UCI move. ``root_move_tokens`` and
    ``root_move_log_probs`` are aligned over the legal root moves in the
    canonical frame of this root (rank-reflected for a black-to-move root).
    """

    root_fen: str
    repetition: int
    mirrored: bool
    tokens: tuple
    thinking_tokens: int
    stop: str
    final_move: str
    root_move_tokens: tuple
    root_move_log_probs: tuple


@dataclass(frozen=True)
class TraceAssessment:
    """Rule-checked facts about a generated trace.

    ``structural`` reports that ``replay`` accepted the token stream and that
    its final move agrees with the writer's decision. The remaining fields are
    derived from the replayed program; on a structural failure they are zero
    or empty because the program could not be rebuilt.
    """

    structural: bool
    error: str | None
    final_move: str | None
    max_depth: int
    max_width: int
    root_width: int
    evaluations: int
    evaluation_revisions: int
    copy_first_move: str | None
    overturned: bool | None


def _framed_move_token(uci, mirrored):
    """Move-vocabulary id of an absolute board move in the trace frame."""
    framed = reflect_uci(uci) if mirrored else uci
    return MOVE + UCI_MOVE_TO_IDX[framed]


def legal_tokens(writer: TraceWriter, final_allowed: bool, cache=None) -> tuple:
    """Token ids the current rule state allows at a thought boundary.

    The set is the vocabulary's evaluation block, RESET, BACK away from the
    root, FINAL when the protocol allows it, and exactly the legal moves of the
    current node in the writer's canonical frame. BOARD and PAD never appear.
    A cache keyed by absolute path avoids recomputing a node's legal set when
    the same path is revisited.
    """
    tokens = [VALUE + index for index in range(PAD - VALUE)]
    tokens.append(RESET)
    if writer.path:
        tokens.append(BACK)
    if final_allowed:
        tokens.append(FINAL)
    path = writer.path
    if cache is not None and path in cache:
        moves = cache[path]
    else:
        moves = tuple(
            _framed_move_token(uci, writer.mirrored) for uci in writer.legal_moves()
        )
        if cache is not None:
            cache[path] = moves
    tokens.extend(moves)
    return tuple(tokens)


class _GenerationRow:
    """Rule state and bookkeeping for one row of a generation batch."""

    def __init__(self, root: RootPosition, protocol: GenerationProtocol):
        self.fen = root.fen
        self.repetition = root.repetition
        self.protocol = protocol
        self.writer = TraceWriter(root.fen, root.repetition)
        board = chess.Board(root.fen)
        self.root_move_by_token = {
            _framed_move_token(move.uci(), self.writer.mirrored): move.uci()
            for move in board.legal_moves
        }
        if not self.root_move_by_token:
            raise ValueError(f"Root has no legal move: {root.fen}")
        self.root_move_tokens = tuple(sorted(self.root_move_by_token))
        self.legal_cache = {(): self.root_move_tokens}
        self.root_move_log_probs = ()
        self.phase = _THOUGHT
        self.thinking = 0
        self.stop = None
        self.final_move = None

    def set_move_distribution(self, logits):
        ids = torch.tensor(self.root_move_tokens, device=logits.device)
        values = logits.float()[ids]
        self.root_move_log_probs = tuple(torch.log_softmax(values, -1).tolist())

    def _check_emitted(self, token):
        if int(self.writer.tokens[-1]) != token:
            raise RuntimeError("TraceWriter emitted a different token")

    def apply(self, token, forced):
        """Consume one selected token, advancing the real rule state."""
        if self.phase == _DONE:
            return
        if self.phase == _ROOT_MOVE:
            move = self.root_move_by_token.get(token)
            if move is None:
                raise RuntimeError("The final decision is not a legal root move")
            self.writer.final(move)
            self._check_emitted(token)
            self.final_move = move
            self.phase = _DONE
            return
        if token == FINAL:
            self.phase = _ROOT_MOVE
            if self.protocol.kind == FIXED:
                self.stop = STOP_BUDGET
            else:
                self.stop = STOP_CAP if forced else STOP_FINAL
            return
        if token == BACK:
            self.writer.back()
        elif token == RESET:
            self.writer.reset()
        elif VALUE <= token < MATE_WIN:
            self.writer.evaluate(value=decode_value_token(token))
        elif MATE_WIN <= token < MATE_LOSS:
            self.writer.evaluate(mate=token - MATE_WIN, win=True)
        elif MATE_LOSS <= token < PAD:
            self.writer.evaluate(mate=token - MATE_LOSS, win=False)
        elif MOVE <= token < VALUE:
            framed = IDX_TO_UCI_MOVE[token - MOVE]
            move = reflect_uci(framed) if self.writer.mirrored else framed
            self.writer.move(move)
        else:
            raise RuntimeError(f"Token {token} is not allowed in a thought")
        self._check_emitted(token)
        self.thinking += 1

    def trace(self) -> GeneratedTrace:
        if self.phase != _DONE or self.stop is None:
            raise RuntimeError("Generation did not reach a final decision")
        return GeneratedTrace(
            self.fen,
            self.repetition,
            self.writer.mirrored,
            tuple(int(token) for token in self.writer.tokens),
            self.thinking,
            self.stop,
            self.final_move,
            self.root_move_tokens,
            self.root_move_log_probs,
        )


def _choose(rows, logits, protocol):
    """Select one legal token per row and record post-FINAL move logits."""
    masks = np.zeros((len(rows), CHESS_VOCAB_SIZE), dtype=bool)
    forced = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        if row.phase == _DONE:
            masks[index, PAD] = True
            continue
        if row.phase == _ROOT_MOVE:
            masks[index, list(row.root_move_tokens)] = True
            continue
        at_limit = row.thinking >= protocol.limit
        forced[index] = at_limit
        final_allowed = at_limit or protocol.kind == AUTONOMOUS
        ids = legal_tokens(row.writer, final_allowed, row.legal_cache)
        masks[index, list(ids)] = True
    allowed = torch.from_numpy(masks).to(logits.device)
    choices = logits.masked_fill(~allowed, -torch.inf).argmax(-1)
    if forced.any():
        forced_mask = torch.from_numpy(forced).to(logits.device)
        choices = torch.where(forced_mask, torch.full_like(choices, FINAL), choices)
    for index, row in enumerate(rows):
        if row.phase == _ROOT_MOVE:
            row.set_move_distribution(logits[index])
    return choices.cpu().numpy(), forced


def _generate_batch(model, roots, protocol):
    device = next(model.parameters()).device
    ids = torch.from_numpy(
        encode_boards([root.fen for root in roots], [root.repetition for root in roots])
    ).to(device)
    rows = [_GenerationRow(root, protocol) for root in roots]
    logits, cache = model.prefill(ids)
    for _ in range(protocol.limit + 2):
        if all(row.phase == _DONE for row in rows):
            break
        choices, forced = _choose(rows, logits, protocol)
        for index, row in enumerate(rows):
            row.apply(int(choices[index]), bool(forced[index]))
        step = np.array(
            [
                PAD if row.phase == _DONE else int(choices[index])
                for index, row in enumerate(rows)
            ],
            dtype=np.int16,
        )
        if all(row.phase == _DONE for row in rows):
            break
        logits, cache = model.decode(
            torch.from_numpy(step).unsqueeze(1).to(device), cache
        )
    if any(row.phase != _DONE for row in rows):
        raise RuntimeError("Generation did not finish within its protocol bound")
    return [row.trace() for row in rows]


@torch.inference_mode()
def generate(model, roots, protocol, batch_size=32):
    """Greedy free generation for frozen roots under one protocol.

    Every step selects the highest-scoring token among the rule-legal set; the
    model is used only through its cached ``prefill``/``decode`` path.
    """
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    roots = list(roots)
    traces = []
    for start in range(0, len(roots), batch_size):
        traces.extend(
            _generate_batch(model, roots[start : start + batch_size], protocol)
        )
    return traces


def assess(trace: GeneratedTrace) -> TraceAssessment:
    """Replay a generated trace and derive its rule-checked facts."""
    try:
        program = replay(trace.tokens, trace.root_fen, trace.repetition)
    except TraceError as error:
        return TraceAssessment(False, str(error), None, 0, 0, 0, 0, 0, None, None)
    structural = program.final_move == trace.final_move
    detail = None if structural else "replayed final move disagrees with the writer"
    evaluations = 0
    revisions = 0
    last_evaluations = {}
    children = {}
    copy_first = None
    for step in program.steps:
        if step.operation == "evaluate":
            evaluations += 1
            token = program.tokens[step.start]
            previous = last_evaluations.get(step.path)
            if previous is not None and previous != token:
                revisions += 1
            last_evaluations[step.path] = token
        elif step.operation == "move":
            if len(step.path) == 1 and copy_first is None:
                copy_first = step.uci
            children.setdefault(step.path[:-1], set()).add(step.path[-1])
    max_depth = max(
        (len(step.path) for step in program.steps if step.operation == "move"),
        default=0,
    )
    max_width = max((len(moves) for moves in children.values()), default=0)
    root_width = len(children.get((), ()))
    overturned = None if copy_first is None else program.final_move != copy_first
    return TraceAssessment(
        structural,
        detail,
        program.final_move,
        max_depth,
        max_width,
        root_width,
        evaluations,
        revisions,
        copy_first,
        overturned,
    )


def move_log_probability(trace: GeneratedTrace, move: str) -> float:
    """Log probability of an absolute legal root move in the final distribution."""
    token = _framed_move_token(move, trace.mirrored)
    try:
        index = trace.root_move_tokens.index(token)
    except ValueError:
        raise ValueError(f"Not a legal root move: {move}") from None
    return trace.root_move_log_probs[index]


def p95_trace_length(lengths) -> int:
    """Nearest-rank 95th percentile of full-trace model token counts."""
    values = np.sort(np.asarray(list(lengths), dtype=np.int64))
    if not len(values):
        raise ValueError("No trace lengths")
    rank = max(1, math.ceil(0.95 * len(values)))
    return int(values[rank - 1])


def trace_length_budget(lengths) -> int:
    """The trace-length bound: p95 rounded up to a multiple of 8, at least 8."""
    return max(8, -(-p95_trace_length(lengths) // 8) * 8)


def budget_points(lengths) -> tuple:
    """Fixed-budget thinking-token points for one teacher's trace lengths.

    The bound is a multiple of 8, so every fixed fraction is an integer.
    """
    bound = trace_length_budget(lengths)
    return tuple(sorted({int(fraction * bound) for fraction in BUDGET_FRACTIONS}))


def budget_grid(lengths_by_teacher) -> tuple:
    """Union of per-teacher budget points, in rising order."""
    points = set()
    for lengths in lengths_by_teacher.values():
        points.update(budget_points(lengths))
    if not points:
        raise ValueError("No teacher trace lengths for a budget grid")
    return tuple(sorted(points))


def model_token_count(tokens, root_fen: str, repetition: int = 1) -> int:
    """Tokens the model generated: everything except BOARD responses.

    A BOARD request stays in the count because the model emitted it; its
    response is external context.
    """
    program = replay(tokens, root_fen, repetition)
    removed = sum(
        step.end - step.start - 1 for step in program.steps if step.operation == "board"
    )
    return len(program.tokens) - removed


def no_tool_token_count(tokens, root_fen: str, repetition: int = 1) -> int:
    """Tokens of the BOARD-free copy: requests and responses both removed."""
    program = replay(tokens, root_fen, repetition)
    removed = sum(
        step.end - step.start for step in program.steps if step.operation == "board"
    )
    return len(program.tokens) - removed


def no_tool_length(tokens) -> int:
    """Length of the BOARD-free copy from the token stream alone.

    This is the complete-trace model token count used for the budget bound:
    BOARD requests and their responses are removed, while navigation,
    evaluations, FINAL, and the final move stay counted. The scan needs no
    rule replay, so a large release can be measured without rebuilding every
    position; ``no_tool_token_count`` is the replay-checked form.
    """
    values = [int(token) for token in tokens]
    index, count = 0, 0
    while index < len(values) and values[index] != PAD:
        if values[index] == BOARD:
            index += 1
            while index < len(values) and values[index] < SIDE:
                index += 1
            index += OBSERVATION_METADATA
        else:
            count += 1
            index += 1
    return count
