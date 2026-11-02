"""Greedy trace generation under the trace grammar's next-token mask."""

from dataclasses import dataclass

import chess
import numpy as np
import torch

from pondera.chess.codec import BOARD_TOKENS_MAX, encode_boards
from pondera.chess.symmetry import reflect_uci
from pondera.chess.vocabulary import (
    CHESS_VOCAB_SIZE,
    FINAL,
    MOVE,
    PAD,
    UCI_MOVE_TO_IDX,
)
from pondera.data.trace import CLAIM_DRAW, TraceError, TraceWriter, replay

FIXED = "fixed"
AUTONOMOUS = "autonomous"
STOP_FINAL = "final"
STOP_BUDGET = "budget"
STOP_CAP = "cap"
_SUFFIX_STEPS = 256


@dataclass(frozen=True)
class GenerationProtocol:
    """A fixed thinking budget or an autonomous safety cap, in tokens."""

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
        return self.budget if self.kind == FIXED else self.safety_cap


def fixed_budget(budget: int) -> GenerationProtocol:
    """Force FINAL after exactly ``budget`` thinking tokens."""
    return GenerationProtocol(FIXED, budget=budget)


def autonomous(safety_cap: int) -> GenerationProtocol:
    """Allow FINAL at any boundary; force it only at ``safety_cap``."""
    return GenerationProtocol(AUTONOMOUS, safety_cap=safety_cap)


@dataclass(frozen=True)
class RootPosition:
    fen: str
    repetition: int = 1
    teacher_width: int = 1

    def __post_init__(self):
        if self.repetition < 1:
            raise ValueError("Root repetition must be positive")
        if self.teacher_width < 1:
            raise ValueError("Teacher width must be positive")


def _effective_teacher_width(root_fen: str, teacher_width: int) -> int:
    return min(teacher_width, chess.Board(root_fen).legal_moves.count())


@dataclass(frozen=True)
class GeneratedTrace:
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
    framed = reflect_uci(uci) if mirrored else uci
    return MOVE + UCI_MOVE_TO_IDX[framed]


def legal_tokens(writer: TraceWriter, final_allowed: bool, cache=None) -> tuple:
    return writer.allowed_tokens(final_allowed, cache)


class _GenerationRow:
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
            ids = legal_tokens(row.writer, False, row.legal_cache)
            masks[index, list(ids)] = True
            continue
        at_limit = row.thinking >= protocol.limit and not row.writer.completing_board
        forced[index] = at_limit
        final_allowed = at_limit or protocol.kind == AUTONOMOUS
        ids = legal_tokens(row.writer, final_allowed, row.legal_cache)
        if not ids:
            ids = legal_tokens(row.writer, True, row.legal_cache)
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
    bound = protocol.limit + BOARD_TOKENS_MAX + 1 + _SUFFIX_STEPS
    for _ in range(bound):
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
    token = _framed_move_token(move, trace.mirrored)
    try:
        index = trace.root_move_tokens.index(token)
    except ValueError:
        raise ValueError(f"Not a legal root move: {move}") from None
    return trace.root_move_log_probs[index]
