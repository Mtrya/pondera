"""Apply shared merge boundaries to stored trace atoms."""

import numpy as np

from pondera.chess.merge import MergeDictionary, MergeEncoder, decode, encode_corpus
from pondera.chess.vocabulary import CHESS_VOCAB_SIZE, DELIM, FINAL
from pondera.data.trace import ROLE_COUNT, ROLE_FINAL_MOVE, derive_trace_roles

# A mixed composite has one next-token loss, rather than one loss per atom.
# Decision-containing composites retain the final-decision diagnostic role.
ROLE_MIXED = ROLE_COUNT

# These atoms and the decision after FINAL must have their own predictions.
STANDALONE_ATOMS = tuple(range(DELIM, CHESS_VOCAB_SIZE))
MERGE_BOUNDARIES = {
    "standalone_atoms": list(STANDALONE_ATOMS),
    "standalone_final_decision": True,
}


def merge_segments(atoms):
    """Isolate the final decision; dictionaries already exclude control atoms."""
    atoms = np.asarray(atoms, dtype=np.int32)
    decisions = np.flatnonzero(atoms == FINAL) + 2
    cuts = np.unique(np.concatenate(([0, len(atoms)], decisions)))
    return [atoms[start:end] for start, end in zip(cuts[:-1], cuts[1:]) if end > start]


def validate_trace_dictionary(dictionary):
    if any(atom >= DELIM for entry in dictionary.entries for atom in entry):
        raise ValueError("A trace composite contains a control atom")


class TraceEncoder(MergeEncoder):
    """Validate a trace dictionary once, then encode within its merge boundaries."""

    def __init__(self, dictionary):
        validate_trace_dictionary(dictionary)
        super().__init__(dictionary)

    def encode(self, atoms):
        if not len(self.dictionary):
            return np.asarray(atoms, dtype=np.int32)
        encode = super().encode
        return np.concatenate(
            [
                encode(segment) if len(segment) > 1 else segment
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
