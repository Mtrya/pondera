"""Sequence trace targets: candidate lines and the final answer as tokens.

A trace is a canonical program over the unified chess vocabulary. Level 0 holds
the answer alone (FINAL, best move, evaluation, END); level 1.5 prefixes one
LINE segment per teacher candidate (LINE, up to ten PV moves, evaluation) and
closes with the same FINAL segment. Moves are stored in the black-to-move
canonical orientation, so a black-to-move root has every PV move rank-reflected;
evaluations keep the teacher's root-player sign. Segment bounds are ordered
[start, end) pairs: the last used entry is the FINAL segment, unused slots are
(-1, -1), and the collator reorders only the leading LINE segments.
"""

from dataclasses import dataclass

from pondera.chess.mapping import IDX_TO_UCI_MOVE, UCI_MOVE_TO_IDX
from pondera.chess.symmetry import reflect_uci
from pondera.chess.tokens import (
    END,
    FINAL,
    LINE,
    MOVE,
    PAD,
    VALUE,
    mate_token,
    value_bin,
)

LEVEL0 = "level0"
LEVEL15 = "level15"
LEVELS = (LEVEL0, LEVEL15)

# Teacher ranks in presented order; MultiPV four leaves the tail absent.
CANDIDATE_RANKS = ("r1", "r2", "r3", "r4")

# PV truncation of a LINE segment, in plies.
PV_LIMIT = 10


@dataclass(frozen=True)
class Segment:
    """One parsed trace segment in the trace orientation."""

    kind: str
    moves: tuple
    evaluation: int
    terminated: bool


def is_move_token(token: int) -> bool:
    return MOVE <= token < MOVE + len(IDX_TO_UCI_MOVE)


def is_eval_token(token: int) -> bool:
    return VALUE <= token < LINE


def depth_snapshot(row, depth=16):
    """The per-depth entry of a teacher release row, or None when absent."""
    for entry in row["unit"]["per_depth"]:
        if entry["d"] == depth:
            return entry
    return None


def evaluation_token(candidate):
    """The candidate's d16 evaluation as a value bin or mate token."""
    if candidate["mate"] is not None:
        return mate_token(abs(candidate["mate"]), candidate["mate"] > 0)
    return VALUE + value_bin(candidate["value"])


def move_token(uci, black):
    """Vocabulary index of a UCI move in the trace's canonical orientation."""
    return MOVE + UCI_MOVE_TO_IDX[reflect_uci(uci) if black else uci]


def absolute_uci(move_tokens, black):
    """Trace-orientation move indices as absolute UCI strings."""
    if black:
        return [reflect_uci(IDX_TO_UCI_MOVE[token - MOVE]) for token in move_tokens]
    return [IDX_TO_UCI_MOVE[token - MOVE] for token in move_tokens]


def candidates(snapshot, limit=len(CANDIDATE_RANKS)):
    """Teacher candidates in rank order, up to the MultiPV width."""
    return [snapshot[key] for key in CANDIDATE_RANKS[:limit] if key in snapshot]


def max_segments(level):
    """Bound on the segment-bound rows of a level."""
    if level not in LEVELS:
        raise ValueError(f"Unknown trace level: {level}")
    return 1 if level == LEVEL0 else 1 + len(CANDIDATE_RANKS)


def encode_trace(snapshot, black, level):
    """Encode one depth snapshot into trace tokens and ordered segment bounds.

    The FINAL segment always uses the rank-1 move and evaluation directly; the
    LINE segments carry the candidates in teacher rank order (training reorders
    them) with PVs truncated to PV_LIMIT plies.
    """
    if level not in LEVELS:
        raise ValueError(f"Unknown trace level: {level}")
    ranked = candidates(snapshot)
    if not ranked:
        raise ValueError("Depth snapshot carries no candidate ranks")
    tokens, bounds = [], []
    if level == LEVEL15:
        for candidate in ranked:
            start = len(tokens)
            tokens.append(LINE)
            tokens.extend(
                move_token(move, black) for move in candidate["pv"][:PV_LIMIT]
            )
            tokens.append(evaluation_token(candidate))
            bounds.append((start, len(tokens)))
    start = len(tokens)
    tokens.extend(
        (
            FINAL,
            move_token(ranked[0]["bm"], black),
            evaluation_token(ranked[0]),
            END,
        )
    )
    bounds.append((start, len(tokens)))
    return tokens, bounds


def parse_trace(tokens):
    """Parse a canonical trace into segments; returns (segments, error tag).

    Errors name the first broken rule, so callers can tell a truncated thought
    from a malformed program. Trailing padding is ignored.
    """
    values = [int(token) for token in tokens]
    while values and values[-1] == PAD:
        values.pop()
    segments, index, error = [], 0, None
    while index < len(values):
        token = values[index]
        if token == LINE:
            index += 1
            moves = []
            while index < len(values) and is_move_token(values[index]):
                moves.append(values[index])
                index += 1
            if not moves:
                error = "line_without_move"
                break
            if index >= len(values):
                error = "truncated"
                break
            if not is_eval_token(values[index]):
                error = "line_missing_evaluation"
                break
            segments.append(Segment("line", tuple(moves), values[index], False))
            index += 1
        elif token == FINAL:
            index += 1
            if index >= len(values) or not is_move_token(values[index]):
                error = "final_without_move"
                break
            move = values[index]
            index += 1
            if index >= len(values) or not is_eval_token(values[index]):
                error = "final_missing_evaluation"
                break
            evaluation = values[index]
            index += 1
            if index >= len(values) or values[index] != END:
                error = "final_missing_end"
                break
            index += 1
            segments.append(Segment("final", (move,), evaluation, True))
            if index != len(values):
                error = "trailing_tokens"
                break
        else:
            error = "unexpected_token"
            break
    if error is None and not any(segment.kind == "final" for segment in segments):
        error = "missing_final"
    return segments, error
