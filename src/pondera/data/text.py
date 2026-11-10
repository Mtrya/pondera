"""Convert the stored SAN transcription into base atoms and training roles."""

import chess
import numpy as np

from pondera.chess.codec import decode_value_token, encode_board, mate_token
from pondera.chess.merge import MergeDictionary, decode, encode_corpus
from pondera.chess.symmetry import TOKEN_MIRROR, reflect_uci
from pondera.chess.vocabulary import (
    BACK,
    BOARD,
    CHESS_VOCAB_SIZE,
    DELIM,
    DEPTH,
    END,
    FINAL,
    MOVE,
    REFUTED,
    RESET,
    UCI_MOVE_TO_IDX,
    VALUE,
    VALUE_BINS,
)
from pondera.data.trace import ROLE_COUNT, ROLE_FINAL_MOVE, derive_trace_roles

# A mixed composite has one next-token loss, rather than one loss per atom.
# Decision-containing composites retain the final-decision diagnostic role.
ROLE_MIXED = ROLE_COUNT

_CONTROLS = {"BACK": BACK, "RESET": RESET, "REFUTED": REFUTED, "END": END}
_VALUES = {
    f"value:{decode_value_token(VALUE + index):+.3f}": VALUE + index
    for index in range(VALUE_BINS)
}

# These atoms and the decision after FINAL must have their own predictions.
STANDALONE_ATOMS = tuple(range(DELIM, CHESS_VOCAB_SIZE))
MERGE_BOUNDARIES = {
    "standalone_atoms": list(STANDALONE_ATOMS),
    "standalone_final_decision": True,
}


def merge_segments(atoms):
    """Split a trace where composites may not cross or absorb a boundary."""
    atoms = np.asarray(atoms, dtype=np.int32)
    controls = np.flatnonzero(atoms >= DELIM)
    decisions = np.flatnonzero(atoms == FINAL) + 2
    cuts = np.unique(
        np.concatenate(([0, len(atoms)], controls, controls + 1, decisions))
    )
    return [atoms[start:end] for start, end in zip(cuts[:-1], cuts[1:]) if end > start]


def validate_trace_dictionary(dictionary):
    if any(atom >= DELIM for entry in dictionary.entries for atom in entry):
        raise ValueError("A trace composite contains a control atom")


def encode_trace(atoms, encoder):
    """Apply the exact encoder independently within each permitted segment."""
    if not len(encoder.dictionary):
        return np.asarray(atoms, dtype=np.int32)
    return np.concatenate(
        [
            encoder.encode(segment) if len(segment) > 1 else segment
            for segment in merge_segments(atoms)
        ]
    )


def encode_traces(traces, dictionary, workers=1):
    """Encode a corpus with the same boundaries used in dictionary fitting."""
    validate_trace_dictionary(dictionary)
    if not len(dictionary):
        return [np.asarray(atoms, dtype=np.int32) for atoms in traces]
    groups = [merge_segments(atoms) for atoms in traces]
    encoded = encode_corpus(
        (segment for group in groups for segment in group), dictionary, workers=workers
    )
    result = []
    position = 0
    for group in groups:
        result.append(np.concatenate(encoded[position : position + len(group)]))
        position += len(group)
    return result


def _position(words, index):
    fen = " ".join(words[index : index + 6])
    if words[index + 6] != "(repetition" or not words[index + 7].endswith(")"):
        raise ValueError("A trace position needs its repetition count")
    repetition = int(words[index + 7][:-1])
    return fen, repetition, index + 8


def trace_atoms(text, *, expected_fen=None, expected_repetition=None):
    """Invert the producer's transcription without teacher or grammar replay.

    SAN uses the absolute current board; stored moves and BOARD observations
    use the root player's frame. FINAL starts its own line from the root even
    when a random presentation ends partway down a thought path.
    """
    words = text.split()
    root_fen, repetition, index = _position(words, 0)
    # Text headers normalize the fullmove counter and saturate observation clocks.
    if expected_fen is not None and not np.array_equal(
        encode_board(root_fen, repetition), encode_board(expected_fen, repetition)
    ):
        raise ValueError("Trace header board differs from the imported row")
    if expected_repetition is not None and repetition != expected_repetition:
        raise ValueError("Trace header repetition differs from the imported row")
    board = chess.Board(root_fen)
    mirrored = board.turn == chess.BLACK
    atoms = []
    final_index = None
    while index < len(words):
        word = words[index]
        index += 1
        if word in _CONTROLS:
            atoms.append(_CONTROLS[word])
            if word == "BACK":
                board.pop()
            elif word == "RESET":
                board = chess.Board(root_fen)
        elif word == "DEPTH":
            int(words[index])
            index += 1
            atoms.append(DEPTH)
        elif word == "BOARD":
            fen, repetition, index = _position(words, index)
            observation = encode_board(fen, repetition)
            if mirrored:
                observation = TOKEN_MIRROR[observation]
            atoms.append(BOARD)
            atoms.extend(observation)
        elif word == "FINAL":
            if final_index is not None:
                raise ValueError("A trace needs one FINAL suffix")
            final_index = len(atoms)
            atoms.append(FINAL)
            board = chess.Board(root_fen)
        elif word == "root":
            continue
        elif word.startswith("value:"):
            atoms.append(_VALUES[word])
        elif word.startswith(("win:", "loss:")):
            side, distance = word.split(":")
            atoms.append(mate_token(int(distance), side == "win"))
        elif word == "CLAIM_DRAW":
            atoms.append(MOVE + UCI_MOVE_TO_IDX["<claim_draw>"])
        else:
            move = board.parse_san(word)
            uci = reflect_uci(move.uci()) if mirrored else move.uci()
            atoms.append(MOVE + UCI_MOVE_TO_IDX[uci])
            board.push(move)
    if (
        final_index is None
        or not atoms
        or atoms[-1] != END
        or final_index + 3 >= len(atoms)
        or not MOVE <= atoms[final_index + 1] < VALUE
        or not VALUE <= atoms[final_index + 2] < VALUE + VALUE_BINS
    ):
        raise ValueError("A complete trace needs a FINAL move, value, and END")
    return np.asarray(atoms, dtype=np.int32)


def encoded_roles(tokens, dictionary=MergeDictionary()):
    """One diagnostic role per stored token, interpreting flat expansions.

    Homogeneous composites retain their atom role; heterogeneous composites
    use MIXED, except a token containing the final decision uses FINAL_MOVE.
    That role locates a token prediction, not an isolated atom prediction.
    """
    if not len(dictionary):
        return derive_trace_roles(tokens)
    tokens = np.asarray(tokens, dtype=np.int64)
    atoms = decode(tokens, dictionary)
    roles = derive_trace_roles(atoms)
    lengths = dictionary.token_lengths[tokens]
    ends = np.cumsum(lengths)
    starts = ends - lengths
    low = np.minimum.reduceat(roles, starts)
    high = np.maximum.reduceat(roles, starts)
    projected = np.where(low == high, low, ROLE_MIXED).astype(np.int8)
    decision = np.flatnonzero(roles == ROLE_FINAL_MOVE)
    projected[np.searchsorted(ends, decision, side="right")] = ROLE_FINAL_MOVE
    return projected
