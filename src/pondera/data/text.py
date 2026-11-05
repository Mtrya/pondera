"""Convert the stored SAN transcription into base atoms and training roles."""

import chess
import numpy as np

from pondera.chess.codec import decode_value_token, encode_board, mate_token
from pondera.chess.merge import MergeDictionary, decode
from pondera.chess.symmetry import TOKEN_MIRROR, reflect_uci
from pondera.chess.vocabulary import (
    BACK,
    BOARD,
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


def _position(words, index):
    fen = " ".join(words[index : index + 6])
    if words[index + 6] != "(repetition" or not words[index + 7].endswith(")"):
        raise ValueError("A trace position needs its repetition count")
    repetition = int(words[index + 7][:-1])
    return fen, repetition, index + 8


def trace_atoms(text):
    """Invert the producer's transcription without teacher or grammar replay.

    SAN uses the absolute current board; stored moves and BOARD observations
    use the root player's frame. FINAL starts its own line from the root even
    when a random presentation ends partway down a thought path.
    """
    words = text.split()
    root_fen, _, index = _position(words, 0)
    board = chess.Board(root_fen)
    mirrored = board.turn == chess.BLACK
    atoms = []
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
    if not atoms or atoms[-1] != END or atoms.count(FINAL) != 1:
        raise ValueError("A complete trace needs one FINAL suffix and END")
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
