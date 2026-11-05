"""Min-cost merge encoding: dictionary contract, optimality, and round trips."""

import json
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from pondera.chess.merge import (
    MAX_MERGE_LENGTH,
    MERGE_ID_BASE,
    Candidate,
    MergeDictionary,
    MergeDictionaryError,
    count_candidates,
    decode,
    encode,
    encode_corpus,
    select_dictionary,
    usage_counts,
)
from pondera.chess.vocabulary import CHESS_VOCAB_SIZE
from pondera.data.config import load_config
from pondera.data.trace import load_sidecar, synthesize_root

FIXTURES = Path(__file__).parent / "fixtures" / "trace_v2"
CASES = ("start_d4", "middlegame_black_d4", "mate_in_1_d4")


def release_rows(cases=CASES):
    """Full rows synthesized by the real pipeline from recorded teacher sidecars.

    The fixtures are the sidecars of an actual depth-4 search, so these are
    the same rows a release shard would carry, only without the release shell.
    """
    config = load_config()
    rows = []
    for case in cases:
        fixture = FIXTURES / case
        root = json.loads((fixture / "root.json").read_text(encoding="utf-8"))
        sidecar = load_sidecar(fixture / "sidecar.jsonl")
        synthesized = synthesize_root(sidecar, root, config)
        rows.extend(
            np.asarray(row.tokens, dtype=np.int64) for row in synthesized.rows.values()
        )
    return rows


def exhaustive_minimum(tokens, dictionary):
    """Fewest tokens by enumerating every segmentation; a second source."""
    tokens = list(tokens)
    best = len(tokens)

    def walk(position, used):
        nonlocal best
        if used >= best:
            return
        if position == len(tokens):
            best = used
            return
        walk(position + 1, used + 1)
        for atoms in dictionary.entries:
            if tuple(tokens[position : position + len(atoms)]) == atoms:
                walk(position + len(atoms), used + 1)

    walk(0, 0)
    return best


def greedy_pairs(tokens, steps):
    """Most frequent adjacent pair merging, keeping flat expansions within 8."""
    sequence = [int(token) for token in tokens]
    expansions = {token: (token,) for token in set(sequence)}
    merges = []
    next_id = MERGE_ID_BASE
    for _ in range(steps):
        pairs = Counter(zip(sequence, sequence[1:]))
        chosen = None
        for pair, _ in sorted(pairs.items(), key=lambda item: (-item[1], item[0])):
            flat = expansions[pair[0]] + expansions[pair[1]]
            if len(flat) <= MAX_MERGE_LENGTH:
                chosen = pair, flat
                break
        if chosen is None:
            break
        (left, right), flat = chosen
        expansions[next_id] = flat
        merges.append(flat)
        merged = []
        index = 0
        while index < len(sequence):
            if (
                index + 1 < len(sequence)
                and sequence[index] == left
                and sequence[index + 1] == right
            ):
                merged.append(next_id)
                index += 2
            else:
                merged.append(sequence[index])
                index += 1
        sequence = merged
        next_id += 1
    return sequence, merges


class CountingTests(unittest.TestCase):
    def test_counts_match_the_hand_derived_windows(self):
        candidates = count_candidates([[1, 2, 1, 2]], min_count=1)
        self.assertEqual(
            {candidate.atoms: candidate.count for candidate in candidates},
            {(1, 2): 2, (2, 1): 1, (1, 2, 1): 1, (2, 1, 2): 1, (1, 2, 1, 2): 1},
        )

    def test_windows_never_cross_a_stream_boundary(self):
        candidates = count_candidates([[1, 2], [2, 1]], min_count=1)
        self.assertEqual(
            {candidate.atoms for candidate in candidates}, {(1, 2), (2, 1)}
        )

    def test_min_count_filters_low_frequency_candidates(self):
        candidates = count_candidates([[1, 2, 1, 2]], min_count=2)
        self.assertEqual([candidate.atoms for candidate in candidates], [(1, 2)])


class EncodingTests(unittest.TestCase):
    def test_tie_takes_the_composite_and_the_longest_entry(self):
        dictionary = MergeDictionary(((0, 0), (0, 0, 0)))
        pair, triple = MERGE_ID_BASE, MERGE_ID_BASE + 1
        # Prefix distances are 0,1,1,1,2. The last position ties between an
        # atom after the triple and the triple after an atom; the composite
        # wins, so the stream is one atom and one triple.
        self.assertEqual(encode([0, 0, 0, 0], dictionary).tolist(), [0, triple])
        self.assertEqual(encode([0, 0, 0], dictionary).tolist(), [triple])
        self.assertEqual(encode([5, 6], MergeDictionary(((5, 6),))).tolist(), [pair])
        self.assertEqual(
            decode(encode([0, 0, 0, 0], dictionary), dictionary).tolist(), [0, 0, 0, 0]
        )

    def test_shorter_match_can_beat_the_longest_match_ending_here(self):
        # Hand-derived: (1,2,3,4)+(5,6,7) is the two-token optimum; the
        # longest entry ending at 7, (3,4,5,6,7), forces three tokens.
        dictionary = MergeDictionary(((1, 2, 3, 4), (3, 4, 5, 6, 7), (5, 6, 7)))
        base = MERGE_ID_BASE
        self.assertEqual(
            encode([1, 2, 3, 4, 5, 6, 7], dictionary).tolist(), [base, base + 2]
        )

    def test_encoding_matches_exhaustive_search_on_a_real_prefix(self):
        row = max(release_rows(("start_d4",)), key=len)
        prefix = row[:20]
        candidates = count_candidates(
            [prefix], min_count=2, max_length=3, per_length_limit=40
        )
        dictionary = MergeDictionary(tuple(candidate.atoms for candidate in candidates))
        self.assertEqual(
            len(encode(prefix, dictionary)), exhaustive_minimum(prefix, dictionary)
        )

    def test_encoding_is_never_worse_than_greedy_pair_merging(self):
        row = max(release_rows(("start_d4",)), key=len)
        sequence, merges = greedy_pairs(row, 20)
        dictionary = MergeDictionary(tuple(dict.fromkeys(merges)))
        encoded = encode(row, dictionary)
        self.assertLessEqual(len(encoded), len(sequence))
        self.assertEqual(decode(encoded, dictionary).tolist(), row.tolist())

    def test_composite_usage_counts_each_stream(self):
        dictionary = MergeDictionary(((1, 2),))
        encodings = encode_corpus([[1, 2, 1, 2], [7, 1, 2]], dictionary)
        self.assertEqual(usage_counts(encodings, dictionary).tolist(), [3])


class SelectionTests(unittest.TestCase):
    def test_selection_drops_candidates_the_encoder_never_uses(self):
        candidates = [Candidate((9, 9), 1000), Candidate((1, 2), 100)]
        dictionary = select_dictionary([[1, 2, 1, 2]], candidates, 8)
        self.assertEqual(dictionary.entries, ((1, 2),))

    def test_selection_respects_the_budget(self):
        candidates = count_candidates([[1, 2, 1, 2, 3, 1, 2, 3]], min_count=1)
        dictionary = select_dictionary([[1, 2, 1, 2, 3, 1, 2, 3]], candidates, 2)
        self.assertLessEqual(len(dictionary), 2)


class DictionaryTests(unittest.TestCase):
    def test_constructor_rejects_out_of_contract_entries(self):
        for entries in [
            ((1,),),
            (tuple(range(9)),),
            ((CHESS_VOCAB_SIZE, 1),),
            ((1, -1),),
            ((1, 2), (1, 2)),
        ]:
            with self.subTest(entries=entries):
                with self.assertRaises(MergeDictionaryError):
                    MergeDictionary(entries)


class ReleaseRoundTripTests(unittest.TestCase):
    def test_round_trip_release_rows_and_compress_them(self):
        rows = release_rows()
        candidates = count_candidates(rows, min_count=2, per_length_limit=64)
        dictionary = select_dictionary(rows, candidates, 64)
        encoded = encode_corpus(rows, dictionary)
        for row, tokens in zip(rows, encoded):
            self.assertEqual(decode(tokens, dictionary).tolist(), row.tolist())
        self.assertLess(
            sum(len(tokens) for tokens in encoded), sum(len(row) for row in rows)
        )


if __name__ == "__main__":
    unittest.main()
