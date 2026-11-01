"""Legal-masked free generation over the sequence trace grammar.

A thought is the trace program of ``pondera.data.trace``: ``<depth>`` opens a
round, moves walk the path, BACK returns to the parent, RESET returns to the
root, value and mate tokens evaluate the current node, and ``<refuted>``
rejects a candidate that fails its reference. ``FINAL`` is the decision point;
the generated label suffix is the evaluation, an optional mate distance, the
best line, and ``<end>``. ``BOARD`` is disabled here, so every generated token
is navigation, a conclusion, the decision, or the label. The generator writes
through ``TraceWriter.apply``; callers validate a trace with ``assess``, which
keeps black-to-move roots in the same canonical rank-reflected frame as
training data. The generation mask and replay share the writer's state machine.

Two protocols bound a thought. A fixed budget masks FINAL until the requested
number of thinking tokens has been generated, then forces FINAL and continues
through the label suffix; a budget of zero is a direct answer. Autonomous
generation allows FINAL at every boundary and stops when the model emits it,
or at a declared safety cap. Thinking tokens are every generated token before
FINAL: depths, moves, BACK, RESET, refutations, and evaluations. The FINAL
delimiter and the whole label suffix never count toward the budget.

The final decision's distribution is the model's post-FINAL distribution
restricted to the legal root moves. Keeping it on the generated trace lets
callers score any legal target move without another forward pass.
"""

import math
from dataclasses import dataclass

import chess
import numpy as np
import torch

from pondera.chess.codec import encode_boards
from pondera.chess.symmetry import reflect_uci
from pondera.chess.vocabulary import (
    BOARD,
    CHESS_VOCAB_SIZE,
    FINAL,
    MOVE,
    PAD,
    SIDE,
    UCI_MOVE_TO_IDX,
)
from pondera.data.trace import (
    CLAIM_DRAW,
    OBSERVATION_METADATA,
    TraceError,
    TraceWriter,
    replay,
)

FIXED = "fixed"
AUTONOMOUS = "autonomous"

# Why generation stopped: the model emitted FINAL, the fixed budget forced it,
# or the autonomous safety cap forced it.
STOP_FINAL = "final"
STOP_BUDGET = "budget"
STOP_CAP = "cap"

# Fixed-budget points as fractions of the trace-length bound.
BUDGET_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)

# The label suffix is bounded by rule terminals, but a pathological best line
# can run long, so generation stops with an error instead of looping forever.
_SUFFIX_STEPS = 256


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
    """One frozen root position, its repetition count, and teacher width."""

    fen: str
    repetition: int = 1
    teacher_width: int = 1

    def __post_init__(self):
        if self.repetition < 1:
            raise ValueError("Root repetition must be positive")
        if self.teacher_width < 1:
            raise ValueError("Teacher width must be positive")


def _effective_teacher_width(root_fen: str, teacher_width: int) -> int:
    """The engine's width: the configured width capped by the legal root moves."""
    return min(teacher_width, chess.Board(root_fen).legal_moves.count())


@dataclass(frozen=True)
class GeneratedTrace:
    """One generated trace and the decision distribution that closed it.

    ``tokens`` is the whole model-generated trace including FINAL, its decision,
    the label evaluation, the best line, and ``<end>``. ``thinking_tokens``
    counts only the tokens before FINAL. ``final_move`` is an absolute UCI move
    or the draw-claim pseudo-move. ``root_move_tokens`` and
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
    teacher_width: int = 1


@dataclass(frozen=True)
class TraceAssessment:
    """Rule-checked facts about a generated trace.

    ``structural`` reports that ``replay`` accepted the token stream and that
    its final move agrees with the writer's decision. The remaining fields are
    derived from the replayed program; on a structural failure they are zero
    or empty because the program could not be rebuilt. ``copy_first_move`` is
    the first round's champion: the argmax of its exact root evaluations when
    that round completed its root scan, otherwise the first root candidate,
    and ``None`` when the trace has no first round.
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

    The writer owns the state machine, so replay, the writer, and this mask
    cannot disagree about visits, refutation references, round boundaries, or
    the FINAL suffix. A cache keyed by absolute path avoids recomputing a
    node's legal set when the same path is revisited.
    """
    return writer.allowed_tokens(final_allowed, cache)


def _generation_tokens(writer: TraceWriter, final_allowed: bool, cache=None) -> tuple:
    """Tokens the generator may choose; BOARD needs a response it cannot give."""
    return tuple(
        token for token in legal_tokens(writer, final_allowed, cache) if token != BOARD
    )


class _GenerationRow:
    """Rule state and bookkeeping for one row of a generation batch."""

    def __init__(self, root: RootPosition, protocol: GenerationProtocol):
        self.fen = root.fen
        self.repetition = root.repetition
        self.teacher_width = root.teacher_width
        self.protocol = protocol
        self.writer = TraceWriter(
            root.fen,
            root.repetition,
            teacher_width=_effective_teacher_width(root.fen, root.teacher_width),
        )
        self.decision_tokens = self.writer.decision_tokens
        self.root_move_tokens = tuple(
            sorted(token for token in self.decision_tokens if token != CLAIM_DRAW)
        )
        self.legal_cache = {}
        self.root_move_log_probs = ()
        self.thinking = 0
        self.stop = None
        self.final_move = None

    def set_move_distribution(self, logits):
        if not self.root_move_tokens:
            self.root_move_log_probs = ()
            return
        ids = torch.tensor(self.root_move_tokens, device=logits.device)
        values = logits.float()[ids]
        self.root_move_log_probs = tuple(torch.log_softmax(values, -1).tolist())

    def _check_emitted(self, token):
        if int(self.writer.tokens[-1]) != token:
            raise RuntimeError("TraceWriter emitted a different token")

    def apply(self, token, forced):
        """Consume one selected token, advancing the real rule state."""
        if self.writer.finished:
            return
        if token == FINAL:
            self.writer.begin_final()
            self._check_emitted(token)
            if self.protocol.kind == FIXED:
                self.stop = STOP_BUDGET
            else:
                self.stop = STOP_CAP if forced else STOP_FINAL
            return
        thought = not self.writer.final_written
        self.writer.apply(token)
        self._check_emitted(token)
        if thought:
            self.thinking += 1
        if self.final_move is None and self.writer.final_move is not None:
            self.final_move = self.writer.final_move

    def trace(self) -> GeneratedTrace:
        if not self.writer.finished or self.stop is None:
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
            self.teacher_width,
        )


def _choose(rows, logits, protocol):
    """Select one legal token per row and record post-FINAL move logits."""
    masks = np.zeros((len(rows), CHESS_VOCAB_SIZE), dtype=bool)
    forced = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        if row.writer.finished:
            masks[index, PAD] = True
            continue
        if row.writer.awaiting_final_move:
            masks[index, list(row.decision_tokens)] = True
            row.set_move_distribution(logits[index])
            continue
        if row.writer.final_written:
            ids = _generation_tokens(row.writer, False, row.legal_cache)
            masks[index, list(ids)] = True
            continue
        at_limit = row.thinking >= protocol.limit
        forced[index] = at_limit
        final_allowed = at_limit or protocol.kind == AUTONOMOUS
        ids = _generation_tokens(row.writer, final_allowed, row.legal_cache)
        if not ids:
            # The grammar allows a root self-evaluation before any candidate,
            # after which FINAL is the only exit. A fixed budget that still
            # masks FINAL must yield rather than stall the row.
            ids = _generation_tokens(row.writer, True, row.legal_cache)
        masks[index, list(ids)] = True
    allowed = torch.from_numpy(masks).to(logits.device)
    choices = logits.masked_fill(~allowed, -torch.inf).argmax(-1)
    if forced.any():
        forced_mask = torch.from_numpy(forced).to(logits.device)
        choices = torch.where(forced_mask, torch.full_like(choices, FINAL), choices)
    return choices.cpu().numpy(), forced


def _generate_batch(model, roots, protocol):
    device = next(model.parameters()).device
    ids = torch.from_numpy(
        encode_boards([root.fen for root in roots], [root.repetition for root in roots])
    ).to(device)
    rows = [_GenerationRow(root, protocol) for root in roots]
    logits, cache = model.prefill(ids)
    for _ in range(protocol.limit + 1 + _SUFFIX_STEPS):
        if all(row.writer.finished for row in rows):
            break
        choices, forced = _choose(rows, logits, protocol)
        for index, row in enumerate(rows):
            row.apply(int(choices[index]), bool(forced[index]))
        step = np.array(
            [
                PAD if row.writer.finished else int(choices[index])
                for index, row in enumerate(rows)
            ],
            dtype=np.int16,
        )
        if all(row.writer.finished for row in rows):
            break
        logits, cache = model.decode(
            torch.from_numpy(step).unsqueeze(1).to(device), cache
        )
    if any(not row.writer.finished for row in rows):
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


def _first_round(program):
    """The first round's champion and the frozen copy-first fallback chain.

    A completed root scan is marked by the round-end root evaluation; its
    champion is the candidate with the highest last exact evaluation. Without
    that evaluation the first written root candidate stands in, and a trace
    with no first round has no copy-first baseline at all.
    """
    round_number = 0
    moves = []
    evaluations = {}
    completed = False
    for step in program.steps:
        if step.operation == "depth":
            round_number += 1
        elif step.operation == "move" and round_number == 1 and len(step.path) == 1:
            if step.uci not in moves:
                moves.append(step.uci)
        elif step.operation == "evaluate" and round_number == 1:
            if len(step.path) == 1:
                evaluations[step.path[0]] = step.evaluation
            elif len(step.path) == 0:
                completed = True
    if round_number == 0:
        return None
    champion = None
    best = None
    if completed:
        for move in moves:
            value = evaluations.get(move)
            if value is None:
                continue
            if best is None or value > best:
                champion, best = move, value
    if champion is None and moves:
        champion = moves[0]
    return champion


def assess(trace: GeneratedTrace) -> TraceAssessment:
    """Replay a generated trace and derive its rule-checked facts."""
    try:
        program = replay(
            trace.tokens,
            trace.root_fen,
            trace.repetition,
            teacher_width=_effective_teacher_width(trace.root_fen, trace.teacher_width),
        )
    except TraceError as error:
        return TraceAssessment(False, str(error), None, 0, 0, 0, 0, 0, None, None)
    structural = program.final_move == trace.final_move
    detail = None if structural else "replayed final move disagrees with the writer"
    evaluations = 0
    revisions = 0
    last_evaluations = {}
    children = {}
    for step in program.steps:
        if step.operation == "evaluate":
            evaluations += 1
            token = program.tokens[step.start]
            previous = last_evaluations.get(step.path)
            if previous is not None and previous != token:
                revisions += 1
            last_evaluations[step.path] = token
        elif step.operation == "move":
            children.setdefault(step.path[:-1], set()).add(step.path[-1])
    max_depth = max(
        (len(step.path) for step in program.steps if step.operation == "move"),
        default=0,
    )
    max_width = max((len(moves) for moves in children.values()), default=0)
    root_width = len(children.get((), ()))
    copy_first = _first_round(program)
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
    """Nearest-rank 95th percentile of complete-trace thinking lengths."""
    values = np.sort(np.asarray(list(lengths), dtype=np.int64))
    if not len(values):
        raise ValueError("No trace lengths")
    rank = max(1, math.ceil(0.95 * len(values)))
    return int(values[rank - 1])


def trace_length_budget(lengths) -> int:
    """The thinking-length bound: p95 rounded up to a multiple of 8, at least 8."""
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


def model_token_count(
    tokens, root_fen: str, repetition: int = 1, teacher_width: int = 1
) -> int:
    """Tokens the model generated: everything except BOARD responses.

    A BOARD request stays in the count because the model emitted it; its
    response is external context.
    """
    program = replay(
        tokens,
        root_fen,
        repetition,
        teacher_width=_effective_teacher_width(root_fen, teacher_width),
    )
    removed = sum(
        step.end - step.start - 1 for step in program.steps if step.operation == "board"
    )
    return len(program.tokens) - removed


def no_tool_token_count(
    tokens, root_fen: str, repetition: int = 1, teacher_width: int = 1
) -> int:
    """Tokens of the BOARD-free copy: requests and responses both removed."""
    program = replay(
        tokens,
        root_fen,
        repetition,
        teacher_width=_effective_teacher_width(root_fen, teacher_width),
    )
    removed = sum(
        step.end - step.start for step in program.steps if step.operation == "board"
    )
    return len(program.tokens) - removed


def no_tool_length(tokens) -> int:
    """Thinking tokens of the BOARD-free copy, ending at FINAL.

    The token budget is the p95 of the pre-FINAL thinking length, so the label
    suffix after FINAL is not counted. BOARD requests and their responses are
    removed. The scan needs no rule replay, so a large release can be measured
    without rebuilding every position; ``no_tool_token_count`` is the
    replay-checked form of the whole-trace count.
    """
    values = [int(token) for token in tokens]
    index, count = 0, 0
    while index < len(values) and values[index] not in (PAD, FINAL):
        if values[index] == BOARD:
            index += 1
            while index < len(values) and values[index] < SIDE:
                index += 1
            index += OBSERVATION_METADATA
        else:
            count += 1
            index += 1
    return count
