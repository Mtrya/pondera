"""Sequence trace targets: candidate lines, board snapshots, and the answer.

A trace is a canonical program over the unified chess vocabulary. Level 0 holds
the answer alone (FINAL, best move, evaluation, END); level 1.5 prefixes one
LINE segment per teacher candidate (LINE, up to ten PV moves, evaluation) and
closes with the same FINAL segment. Levels s2e and s2r3 additionally interleave
explicit board snapshots inside every LINE: s2e at the end of each line, s2r3
after every third move and at the end.

Moves are stored in the black-to-move canonical orientation, so a black-to-move
root has every PV move rank-reflected; evaluations keep the teacher's
root-player sign. A snapshot is a run of board atoms in a1..h8 scan order (the
same ids a board prefix uses) followed by six metadata atoms (side, castling,
en passant) without counters, and it is emitted in the same canonical frame:
pieces rank-mirrored and color-swapped for a black-to-move root, side atom in
the root frame's naming, which the mirror leaves unchanged. Snapshots carry
rule truth replayed with python-chess, so their side, castling, and en-passant
fields follow the pushed PV rather than the root.

Segment bounds are ordered [start, end) pairs: the last used entry is the FINAL
segment, unused slots are (-1, -1), and the collator reorders only the leading
LINE segments, each whole line with its snapshots.
"""

from dataclasses import dataclass

import chess

from pondera.chess.codec import mate_token, value_bin
from pondera.chess.symmetry import reflect_uci
from pondera.chess.vocabulary import (
    BOARD_PIECES,
    BOARD_TOKENS,
    CASTLING,
    EN_PASSANT,
    END,
    FINAL,
    HALFMOVE,
    IDX_TO_UCI_MOVE,
    LINE,
    MOVE,
    PAD,
    SIDE,
    UCI_MOVE_TO_IDX,
    VALUE,
)

LEVEL0 = "level0"
LEVEL15 = "level15"
S2E = "s2e"
S2R3 = "s2r3"
LEVELS = (LEVEL0, LEVEL15, S2E, S2R3)

# Teacher ranks in presented order; MultiPV four leaves the tail absent.
CANDIDATE_RANKS = ("r1", "r2", "r3", "r4")

# PV truncation of a LINE segment, in plies.
PV_LIMIT = 10

# Snapshot cadence of the s2r3 level, in plies.
SNAPSHOT_INTERVAL = 3

# Metadata fields of one snapshot run, in order after the piece tokens.
SNAPSHOT_METADATA = 6


@dataclass(frozen=True)
class Segment:
    """One parsed trace segment in the trace orientation.

    snapshots holds (moves_before, token run) pairs for the board snapshots
    carried by the segment, in the order they were emitted.
    """

    kind: str
    moves: tuple
    evaluation: int
    terminated: bool
    snapshots: tuple = ()


@dataclass(frozen=True)
class Snapshot:
    """Decoded board snapshot: occupied squares plus its metadata fields."""

    pieces: tuple
    side_black: bool
    castling: tuple
    ep: int


def is_move_token(token: int) -> bool:
    return MOVE <= token < MOVE + len(IDX_TO_UCI_MOVE)


def is_eval_token(token: int) -> bool:
    return VALUE <= token < PAD


def is_board_token(token: int) -> bool:
    return 0 <= token < BOARD_TOKENS


def is_snapshot_metadata(token: int) -> bool:
    """A board-metadata id, only legal inside a snapshot's trailing block."""
    return SIDE <= token <= EN_PASSANT + 64


def piece_code(piece) -> int:
    """Board-encoding piece code of a python-chess piece (0..11, no empty)."""
    return (piece.piece_type - 1) + (0 if piece.color else 6)


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


def canonical_board(board, mirror):
    """The position in the trace's canonical frame (ranks flipped on a mirror).

    python-chess's mirror also flips the side to move; the snapshot's side token
    is taken from the root frame separately, so the flip is irrelevant here.
    """
    return board.mirror() if mirror else board.copy()


def snapshot_pieces(board):
    """Occupied (square, piece code) pairs of a position in a1..h8 scan order."""
    return tuple(
        sorted(
            (square, piece_code(piece)) for square, piece in board.piece_map().items()
        )
    )


def snapshot_run(board, side_black):
    """Snapshot token run of a position already in the canonical frame.

    side_black is the side to move in the root frame, which the canonical mirror
    leaves unchanged, matching the board prefix's side-token convention.
    """
    tokens = [square * BOARD_PIECES + piece for square, piece in snapshot_pieces(board)]
    tokens.append(SIDE + side_black)
    rights = (
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )
    tokens.extend(CASTLING + 2 * index + int(flag) for index, flag in enumerate(rights))
    tokens.append(EN_PASSANT + (board.ep_square if board.ep_square is not None else 64))
    return tokens


def snapshot_of(board, mirror):
    """Snapshot token run of a true position in the trace's canonical frame."""
    return snapshot_run(canonical_board(board, mirror), board.turn == chess.BLACK)


def decode_snapshot(tokens) -> Snapshot:
    """Decode one snapshot token run; raises when the run is malformed.

    Metadata atoms are self-contained and identified by their ids, so their
    order inside the six-atom block is not significant.
    """
    pieces, index = [], 0
    while index < len(tokens) and is_board_token(tokens[index]):
        pieces.append(divmod(int(tokens[index]), BOARD_PIECES))
        index += 1
    if not pieces:
        raise ValueError("A snapshot needs at least one piece atom")
    if any(
        square >= next_square
        for (square, _), (next_square, _) in zip(pieces, pieces[1:])
    ):
        raise ValueError("Snapshot squares must increase in scan order")
    metadata = [int(token) for token in tokens[index : index + SNAPSHOT_METADATA]]
    if len(metadata) != SNAPSHOT_METADATA or not all(
        is_snapshot_metadata(token) for token in metadata
    ):
        raise ValueError("Snapshot metadata is truncated")
    side = [token for token in metadata if token in (SIDE, SIDE + 1)]
    castling_tokens = [token for token in metadata if CASTLING <= token < EN_PASSANT]
    ep_tokens = [token for token in metadata if EN_PASSANT <= token < HALFMOVE]
    if len(side) != 1:
        raise ValueError("Invalid snapshot side token")
    flags = {}
    for token in castling_tokens:
        field, flag = divmod(token - CASTLING, 2)
        if field in flags:
            raise ValueError("Invalid snapshot castling token")
        flags[field] = flag
    if len(flags) != 4 or len(ep_tokens) != 1:
        raise ValueError("Invalid snapshot castling token")
    return Snapshot(
        tuple(pieces),
        side[0] == SIDE + 1,
        tuple(flags[field] for field in range(4)),
        ep_tokens[0] - EN_PASSANT,
    )


def snapshot_positions(level, moves):
    """Move counts after which a LINE segment of this level carries a snapshot."""
    if level == S2E:
        return (moves,) if moves else ()
    return tuple(range(SNAPSHOT_INTERVAL, moves + 1, SNAPSHOT_INTERVAL)) + (
        (moves,) if moves % SNAPSHOT_INTERVAL else ()
    )


def encode_trace(snapshot, black, level, board=None):
    """Encode one depth snapshot into trace tokens and ordered segment bounds.

    The FINAL segment always uses the rank-1 move and evaluation directly; the
    LINE segments carry the candidates in teacher rank order (training reorders
    them) with PVs truncated to PV_LIMIT plies. Levels s2e and s2r3 additionally
    interleave board snapshots and require the root position as a python-chess
    board, which supplies the rule truth for every snapshot point.
    """
    if level not in LEVELS:
        raise ValueError(f"Unknown trace level: {level}")
    if level in (S2E, S2R3) and board is None:
        raise ValueError("Snapshot levels need the root position")
    ranked = candidates(snapshot)
    if not ranked:
        raise ValueError("Depth snapshot carries no candidate ranks")
    tokens, bounds = [], []
    if level != LEVEL0:
        for candidate in ranked:
            moves = [move_token(move, black) for move in candidate["pv"][:PV_LIMIT]]
            start = len(tokens)
            tokens.append(LINE)
            if level in (S2E, S2R3):
                if not moves:
                    raise ValueError("Snapshot levels need at least one PV move")
                replay = canonical_board(board, black)
                due = set(snapshot_positions(level, len(moves)))
                for number, move in enumerate(moves, 1):
                    tokens.append(move)
                    replay.push_uci(IDX_TO_UCI_MOVE[move - MOVE])
                    if number in due:
                        side_black = black if number % 2 == 0 else not black
                        tokens.extend(snapshot_run(replay, side_black))
            else:
                tokens.extend(moves)
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


def read_snapshot(values, index):
    """Read one snapshot run at index; returns (run, next index, error tag)."""
    start, squares = index, []
    while index < len(values) and is_board_token(values[index]):
        squares.append(values[index] // BOARD_PIECES)
        index += 1
    if not squares:
        return (), index, "snapshot_without_pieces"
    if any(square >= next_square for square, next_square in zip(squares, squares[1:])):
        return (), index, "snapshot_square_order"
    metadata = values[index : index + SNAPSHOT_METADATA]
    if len(metadata) < SNAPSHOT_METADATA or not all(
        is_snapshot_metadata(token) for token in metadata
    ):
        return (), index, "snapshot_truncated"
    side = [token for token in metadata if token in (SIDE, SIDE + 1)]
    if len(side) != 1:
        return (), index, "snapshot_side"
    fields = [
        (token - CASTLING) // 2 for token in metadata if CASTLING <= token < EN_PASSANT
    ]
    ep = [token for token in metadata if EN_PASSANT <= token < HALFMOVE]
    if len(fields) != 4 or len(set(fields)) != 4 or len(ep) != 1:
        return (), index, "snapshot_castling"
    return (
        tuple(values[start : index + SNAPSHOT_METADATA]),
        index + SNAPSHOT_METADATA,
        None,
    )


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
            moves, snapshots = [], []
            while index < len(values):
                if is_move_token(values[index]):
                    moves.append(values[index])
                    index += 1
                elif is_board_token(values[index]):
                    run, index, error = read_snapshot(values, index)
                    if error is not None:
                        break
                    snapshots.append((len(moves), run))
                else:
                    break
            if error is not None:
                break
            if not moves:
                error = "line_without_move"
                break
            if index >= len(values):
                error = "truncated"
                break
            if not is_eval_token(values[index]):
                error = "line_missing_evaluation"
                break
            segments.append(
                Segment("line", tuple(moves), values[index], False, tuple(snapshots))
            )
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
