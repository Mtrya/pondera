"""Apply shared merge boundaries to stored trace atoms."""

import numpy as np

from pondera.chess.merge import MergeEncoder, encode_corpus
from pondera.chess.vocabulary import CHESS_VOCAB_SIZE, DELIM, FINAL

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
