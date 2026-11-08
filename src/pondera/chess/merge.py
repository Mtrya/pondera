"""Min-cost merging of atom streams into composite trace tokens.

The frozen vocabulary is the atom layer. A merge dictionary adds composite
tokens whose ids continue past it (``CHESS_VOCAB_SIZE``, ``CHESS_VOCAB_SIZE +
1``, ...), while PAD retains its existing atom id. Every composite expands to a flat
run of 2..8 atoms. Dataset metadata and model checkpoints store these flat
expansions in composite-ID order.

Counting follows the counting-and-filtering idea of CNF-MCE
(arXiv:2610.01127) rather than any particular implementation: every atom
substring of length 2..8 is a candidate with an occurrence count, and the
counted pool is what selection may admit. Counting never lets a window cross
a stream boundary.

Selection is driven by the encoder rather than by raw frequency. A substring
that only ties the shorter pieces it contains looks valuable when counted
alone, and a long entry absorbs the uses of the shorter entries it overlaps.
Each round therefore encodes the corpus with the current dictionary, counts
the adjacent token pairs that encoding actually produces, and admits the most
frequent pair merges whose flat expansion is a counted candidate; composites
the encoder never chose are dropped first. The rounds trade adaptivity for
encoder passes: one pair per round is BPE's greedy merge, and larger blocks
do nearly as well with far fewer passes.

Encoding is a shortest-path segmentation of one atom stream into atoms and
dictionary entries under a unit cost per emitted token, so the optimum is the
fewest tokens. Every entry ending at a position is a candidate: prefix
distances are not monotone, because an entry can jump over atoms and make a
later prefix cheaper than an earlier one, so the longest entry ending at a
position is not always the best one. Ties prefer the composite and then the
longer entry, which together with a fixed candidate ordering makes the
encoding a function of the atom stream and the dictionary alone.
"""

import multiprocessing
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from operator import index as _index

import numpy as np

from pondera.chess.vocabulary import CHESS_VOCAB_SIZE

MIN_MERGE_LENGTH = 2
MAX_MERGE_LENGTH = 8
MERGE_ID_BASE = CHESS_VOCAB_SIZE

# Windows are packed into 12-bit atom fields for matching and counting; the
# vocabulary has far fewer atoms, and five fields fit one signed 64-bit
# integer, so longer windows use a two-word key.
_ATOM_BITS = 12
_FIELDS_PER_WORD = 4
_SINGLE_WORD_LENGTH = 5
_FIELD_MASK = (1 << _ATOM_BITS) - 1
# Streams are joined with a separator so no window spans two streams.
_SEPARATOR = -1

_PAIR_DTYPE = np.dtype([("high", "<i8"), ("low", "<i8")])
_VOID_DTYPE = np.dtype((np.void, _PAIR_DTYPE.itemsize))


class MergeDictionaryError(ValueError):
    """A dictionary that violates the flat composite-token contract."""


@dataclass(frozen=True)
class Candidate:
    """One atom substring and its occurrence count in the counted corpus."""

    atoms: tuple
    count: int

    @property
    def savings(self) -> int:
        """Tokens saved by using the candidate once instead of its atoms."""
        return self.count * (len(self.atoms) - 1)


@dataclass(frozen=True)
class MergeDictionary:
    """Composite tokens in id order; entry ``i`` has id ``MERGE_ID_BASE + i``.

    Validation is the format contract: 2..8 atom ids per entry, ids below
    ``MERGE_ID_BASE``, and no duplicate expansion.
    """

    entries: tuple = ()

    def __post_init__(self):
        entries = tuple(tuple(_index(atom) for atom in atoms) for atoms in self.entries)
        seen = set()
        for atoms in entries:
            if not MIN_MERGE_LENGTH <= len(atoms) <= MAX_MERGE_LENGTH:
                raise MergeDictionaryError(
                    f"A composite needs {MIN_MERGE_LENGTH} to {MAX_MERGE_LENGTH} atoms"
                )
            for atom in atoms:
                if not 0 <= atom < MERGE_ID_BASE:
                    raise MergeDictionaryError(f"Not an atom id: {atom}")
            if atoms in seen:
                raise MergeDictionaryError(f"Duplicate composite: {atoms}")
            seen.add(atoms)
        object.__setattr__(self, "entries", entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    @property
    def ids(self) -> range:
        """The composite ids, in entry order."""
        return range(MERGE_ID_BASE, MERGE_ID_BASE + len(self.entries))

    def token_id(self, index: int) -> int:
        return MERGE_ID_BASE + index

    def expansion(self, token: int) -> tuple:
        index = token - MERGE_ID_BASE
        if not 0 <= index < len(self.entries):
            raise MergeDictionaryError(
                f"Not a composite token of this dictionary: {token}"
            )
        return self.entries[index]

    @cached_property
    def token_lengths(self):
        """Flat expansion lengths indexed by either an atom or composite ID."""
        return np.asarray(
            [1] * MERGE_ID_BASE + [len(atoms) for atoms in self.entries], dtype=np.int8
        )


def _streams(sequences) -> list:
    streams = []
    for sequence in sequences:
        stream = np.asarray(sequence, dtype=np.int64)
        if stream.ndim != 1:
            raise ValueError("A token stream is one-dimensional")
        if stream.size:
            low, high = int(stream.min()), int(stream.max())
            if low < 0 or high >= MERGE_ID_BASE:
                raise ValueError("Streams hold atom ids in [0, MERGE_ID_BASE)")
        streams.append(stream)
    return streams


def _streams32(sequences) -> list:
    """Validated atom streams in the compact dtype counting joins together."""
    streams = []
    for sequence in sequences:
        stream = np.asarray(sequence, dtype=np.int32)
        if stream.ndim != 1:
            raise ValueError("A token stream is one-dimensional")
        if stream.size:
            low, high = int(stream.min()), int(stream.max())
            if low < 0 or high >= MERGE_ID_BASE:
                raise ValueError("Streams hold atom ids in [0, MERGE_ID_BASE)")
        streams.append(stream)
    return streams


def _spans(masses, span_mass) -> list:
    """Contiguous [start, stop) spans each holding at most ``span_mass`` mass."""
    spans = []
    start, mass = 0, 0
    for index, value in enumerate(masses):
        if mass and mass + value > span_mass:
            spans.append((start, index))
            start, mass = index, 0
        mass += value
    if start < len(masses):
        spans.append((start, len(masses)))
    return spans


def _pack(windows) -> np.ndarray:
    """Pack each row of a (n, length) window view into an exact key."""
    length = windows.shape[1]
    if length <= _SINGLE_WORD_LENGTH:
        key = windows[:, 0].astype(np.int64)
        for column in range(1, length):
            key = (key << _ATOM_BITS) | windows[:, column]
        return key
    high = windows[:, 0].astype(np.int64)
    for column in range(1, _FIELDS_PER_WORD):
        high = (high << _ATOM_BITS) | windows[:, column]
    low = windows[:, _FIELDS_PER_WORD].astype(np.int64)
    for column in range(_FIELDS_PER_WORD + 1, length):
        low = (low << _ATOM_BITS) | windows[:, column]
    pair = np.empty(len(windows), dtype=_PAIR_DTYPE)
    pair["high"] = high
    pair["low"] = low
    return pair.view(_VOID_DTYPE)


def _unpack(key, length: int) -> tuple:
    if length <= _SINGLE_WORD_LENGTH:
        value = int(key)
        return tuple(
            (value >> (_ATOM_BITS * shift)) & _FIELD_MASK
            for shift in range(length - 1, -1, -1)
        )
    raw = key.tobytes()
    high = int.from_bytes(raw[:8], "little")
    low = int.from_bytes(raw[8:], "little")
    fields = [
        (high >> (_ATOM_BITS * shift)) & _FIELD_MASK
        for shift in range(_FIELDS_PER_WORD - 1, -1, -1)
    ]
    fields.extend(
        (low >> (_ATOM_BITS * shift)) & _FIELD_MASK
        for shift in range(length - _FIELDS_PER_WORD - 1, -1, -1)
    )
    return tuple(fields)


def _join(streams) -> np.ndarray:
    total = sum(len(stream) for stream in streams) + max(len(streams) - 1, 0)
    body = np.empty(total, dtype=np.int32)
    position = 0
    for index, stream in enumerate(streams):
        if index:
            body[position] = _SEPARATOR
            position += 1
        body[position : position + len(stream)] = stream
        position += len(stream)
    return body


def _count_chunk(streams, max_length, forbidden) -> dict:
    """Per-length (unique keys, counts) counted inside one chunk of streams."""
    body = _join(streams)
    blocked = np.isin(body, list(forbidden)) if forbidden else None
    result = {}
    for length in range(MIN_MERGE_LENGTH, max_length + 1):
        if body.size < length:
            break
        windows = np.lib.stride_tricks.sliding_window_view(body, length)
        inside = windows.min(axis=1) >= 0
        if blocked is not None:
            inside &= ~np.lib.stride_tricks.sliding_window_view(blocked, length).any(
                axis=1
            )
        if not inside.any():
            continue
        keys = _pack(windows)[inside]
        result[length] = np.unique(keys, return_counts=True)
    return result


_COUNT_STREAMS = None
_COUNT_MAX_LENGTH = MAX_MERGE_LENGTH
_COUNT_FORBIDDEN = ()


def _count_pool_init(max_length, forbidden):
    global _COUNT_MAX_LENGTH, _COUNT_FORBIDDEN
    _COUNT_MAX_LENGTH = max_length
    _COUNT_FORBIDDEN = forbidden


def _count_span(span):
    start, stop = span
    return _count_chunk(_COUNT_STREAMS[start:stop], _COUNT_MAX_LENGTH, _COUNT_FORBIDDEN)


def count_candidates(
    sequences,
    *,
    min_count: int = 2,
    max_length: int = MAX_MERGE_LENGTH,
    per_length_limit=None,
    forbidden=(),
    workers: int = 1,
    chunk_atoms: int = 1 << 26,
) -> list:
    """Occurrence counts of every atom substring of length 2..max_length.

    A window never crosses a stream boundary, and a window containing a
    forbidden atom is never counted. Streams are counted in chunks of about
    ``chunk_atoms`` atoms to bound memory; chunk counts merge exactly because
    no window crosses a boundary. With ``workers`` above one, forked worker
    processes count the chunks, inheriting the streams copy-on-write. Only
    candidates occurring at least ``min_count`` times survive;
    ``per_length_limit`` further caps each length, keeping the highest-saving
    candidates. The result is sorted by potential savings, then count, then
    atoms, so the caller's pool is deterministic.
    """
    if min_count < 1:
        raise ValueError("min_count must be positive")
    if not MIN_MERGE_LENGTH <= max_length <= MAX_MERGE_LENGTH:
        raise ValueError(
            f"max_length must lie in [{MIN_MERGE_LENGTH}, {MAX_MERGE_LENGTH}]"
        )
    if per_length_limit is not None and per_length_limit < 1:
        raise ValueError("per_length_limit must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if chunk_atoms < 1:
        raise ValueError("chunk_atoms must be positive")
    forbidden = tuple(_index(atom) for atom in forbidden)
    for atom in forbidden:
        if not 0 <= atom < MERGE_ID_BASE:
            raise ValueError(f"Not an atom id: {atom}")
    streams = _streams32(sequences)
    spans = _spans([stream.size for stream in streams], chunk_atoms)
    if workers > 1 and len(spans) > 1:
        global _COUNT_STREAMS
        _COUNT_STREAMS = streams
        try:
            context = multiprocessing.get_context("fork")
            with context.Pool(
                workers, initializer=_count_pool_init, initargs=(max_length, forbidden)
            ) as pool:
                chunks = pool.map(_count_span, spans)
        finally:
            _COUNT_STREAMS = None
    else:
        chunks = [
            _count_chunk(streams[start:stop], max_length, forbidden)
            for start, stop in spans
        ]
    found = []
    for length in range(MIN_MERGE_LENGTH, max_length + 1):
        portions = [chunk[length] for chunk in chunks if length in chunk]
        if not portions:
            continue
        keys = np.concatenate([portion[0] for portion in portions])
        counts = np.concatenate([portion[1] for portion in portions])
        order = np.argsort(keys, kind="stable")
        keys, counts = keys[order], counts[order]
        boundaries = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
        keys, counts = keys[boundaries], np.add.reduceat(counts, boundaries)
        order = np.argsort(-counts, kind="stable")
        order = order[counts[order] >= min_count]
        if per_length_limit is not None:
            order = order[:per_length_limit]
        found.extend(
            Candidate(_unpack(keys[index], length), int(counts[index]))
            for index in order
        )
    found.sort(
        key=lambda candidate: (-candidate.savings, -candidate.count, candidate.atoms)
    )
    return found


def _entry_tables(dictionary: MergeDictionary) -> list:
    """Per length, sorted match keys and the composite ids they map to."""
    grouped = {}
    for index, atoms in enumerate(dictionary.entries):
        grouped.setdefault(len(atoms), []).append((atoms, MERGE_ID_BASE + index))
    tables = []
    for length in sorted(grouped):
        rows = np.asarray([atoms for atoms, _ in grouped[length]], dtype=np.int64)
        ids = np.asarray([token for _, token in grouped[length]], dtype=np.int32)
        keys = _pack(rows)
        order = np.argsort(keys, kind="stable")
        tables.append((length, keys[order], ids[order]))
    return tables


def _match(stream: np.ndarray, tables) -> tuple:
    """Every dictionary entry ending at each position of one stream.

    Returns ``(size + 1, slots)`` entry lengths and composite ids (padding 0
    and -1) plus the number of entries ending at each position. Columns fill
    in ascending length order, and no window crosses a stream boundary.
    """
    size = stream.size
    length_at = np.zeros((size + 1, len(tables)), dtype=np.int8)
    token_at = np.full((size + 1, len(tables)), -1, dtype=np.int32)
    count_at = np.zeros(size + 1, dtype=np.int8)
    for length, keys, ids in tables:
        if stream.size < length:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(stream, length)
        keys_of_stream = _pack(windows)
        position = np.searchsorted(keys, keys_of_stream)
        clipped = np.minimum(position, keys.size - 1)
        hit = keys[clipped] == keys_of_stream
        starts = np.flatnonzero(hit)
        if starts.size == 0:
            continue
        ends = starts + length
        columns = count_at[ends]
        length_at[ends, columns] = length
        token_at[ends, columns] = ids[clipped[starts]]
        count_at[ends] += 1
    return length_at, token_at, count_at


def _encode_stream(stream, length_at, token_at, count_at) -> np.ndarray:
    """Minimum-token segmentation of one stream; ties take the composite.

    Among composites of equal cost the longest wins, matching the previous
    longest-per-end rule whenever that rule found an optimum.
    """
    size = stream.size
    lengths = length_at.tolist()
    tokens = token_at.tolist()
    counts = count_at.tolist()
    distance = [0] * (size + 1)
    choice = [-1] * (size + 1)
    chosen_length = [0] * (size + 1)
    for index in range(1, size + 1):
        best = distance[index - 1] + 1
        token = -1
        length = 0
        for slot in range(counts[index]):
            entry_length = lengths[index][slot]
            candidate = distance[index - entry_length] + 1
            if candidate < best or (candidate == best and entry_length > length):
                best = candidate
                token = tokens[index][slot]
                length = entry_length
        distance[index] = best
        choice[index] = token
        chosen_length[index] = length
    encoded = []
    index = size
    while index:
        token = choice[index]
        if token < 0:
            encoded.append(int(stream[index - 1]))
            index -= 1
        else:
            encoded.append(token)
            index -= chosen_length[index]
    encoded.reverse()
    return np.asarray(encoded, dtype=np.int32)


class MergeEncoder:
    """Reuse dictionary match tables across independently encoded streams."""

    def __init__(self, dictionary: MergeDictionary):
        self.dictionary = dictionary
        self.tables = _entry_tables(dictionary)

    def encode(self, tokens):
        stream = _streams((tokens,))[0]
        if not self.tables:
            return stream.astype(np.int32)
        return _encode_stream(stream, *_match(stream, self.tables))


_WORKER_STREAMS = None
_WORKER_ENCODER = None


def _encode_pool_init(dictionary):
    global _WORKER_ENCODER
    _WORKER_ENCODER = MergeEncoder(dictionary)


def _encode_span(span):
    start, stop = span
    encoder = _WORKER_ENCODER
    return [encoder.encode(_WORKER_STREAMS[index]) for index in range(start, stop)]


def encode_corpus(sequences, dictionary: MergeDictionary, workers: int = 1) -> list:
    """Encode every atom stream with the dictionary, in input order.

    With ``workers`` above one, forked worker processes encode atom-balanced
    spans of the stream list, inheriting the streams copy-on-write; the
    result is identical to sequential encoding.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    streams = list(sequences)
    masses = [len(stream) for stream in streams]
    if workers == 1 or len(streams) < 2 or sum(masses) < 1 << 20:
        encoder = MergeEncoder(dictionary)
        return [encoder.encode(stream) for stream in streams]
    global _WORKER_STREAMS
    _WORKER_STREAMS = streams
    try:
        spans = _spans(masses, max(1, -(-sum(masses) // (workers * 4))))
        context = multiprocessing.get_context("fork")
        with context.Pool(
            workers, initializer=_encode_pool_init, initargs=(dictionary,)
        ) as pool:
            chunks = pool.map(_encode_span, spans)
    finally:
        _WORKER_STREAMS = None
    return [encoded for chunk in chunks for encoded in chunk]


def encode(tokens, dictionary: MergeDictionary) -> np.ndarray:
    """Encode one atom stream with the minimum number of tokens."""
    return MergeEncoder(dictionary).encode(tokens)


def usage_counts(encodings, dictionary: MergeDictionary) -> np.ndarray:
    """How often each composite was used across the encoded streams."""
    counts = np.zeros(len(dictionary), dtype=np.int64)
    if not len(dictionary):
        return counts
    for encoded in encodings:
        values = np.asarray(encoded, dtype=np.int64)
        composite = values[values >= MERGE_ID_BASE] - MERGE_ID_BASE
        counts += np.bincount(composite, minlength=len(dictionary))
    return counts


def decode(tokens, dictionary: MergeDictionary) -> np.ndarray:
    """Expand composite ids; atom ids pass through unchanged."""
    atoms = []
    for token in np.asarray(tokens, dtype=np.int64).tolist():
        if token >= MERGE_ID_BASE:
            atoms.extend(dictionary.expansion(token))
        elif token < 0:
            raise MergeDictionaryError(f"Not a token id: {token}")
        else:
            atoms.append(token)
    return np.asarray(atoms, dtype=np.int64)


def _unique_candidates(candidates) -> list:
    seen = set()
    pool = []
    for candidate in candidates:
        if candidate.atoms in seen:
            continue
        if not MIN_MERGE_LENGTH <= len(candidate.atoms) <= MAX_MERGE_LENGTH:
            raise ValueError("A candidate needs 2 to 8 atoms")
        seen.add(candidate.atoms)
        pool.append(candidate)
    return pool


def _expand(token: int, expansions) -> tuple:
    return (token,) if token < MERGE_ID_BASE else expansions[token]


def _merge_counts(encodings, expansions, pool) -> list:
    """Rank the adjacent token pairs of an encoding by realized occurrences.

    Merging one pair replaces two tokens by one, so every occurrence saves
    exactly one token; raw pair counts rank the block, with the atoms as the
    deterministic tie-break.
    """
    counts = Counter()
    for encoded in encodings:
        values = encoded.tolist()
        counts.update(zip(values, values[1:]))
    merges = {}
    for (left, right), count in counts.items():
        atoms = _expand(left, expansions) + _expand(right, expansions)
        if len(atoms) > MAX_MERGE_LENGTH or atoms not in pool:
            continue
        merges[atoms] = merges.get(atoms, 0) + count
    return sorted(merges, key=lambda atoms: (-merges[atoms], atoms))


def select_dictionary(
    sequences,
    candidates,
    budget: int,
    *,
    block: int = 64,
    max_rounds: int = 100_000,
    workers: int = 1,
) -> MergeDictionary:
    """Grow at most ``budget`` composites from counted candidates.

    Each round encodes the streams with the current dictionary, drops
    composites the encoder never chose, then admits up to ``block`` of the
    most frequent pair merges of that encoding. Only counted substrings can be
    admitted and no merge may exceed eight atoms, so the pool from
    `count_candidates` is the vocabulary's ceiling. The loop stops at the
    budget, when the encoder produces no new merge, or after ``max_rounds``; a
    final pass drops the entries the encoder did not choose. The entries keep
    their admission order, most valuable first. Encoding passes use
    ``workers`` forked worker processes when above one.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if block < 1:
        raise ValueError("block must be positive")
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    pool = {candidate.atoms for candidate in _unique_candidates(candidates)}
    if budget == 0 or not pool:
        return MergeDictionary(())
    streams = _streams(sequences)
    entries = []
    for _ in range(max_rounds):
        if len(entries) >= budget:
            break
        dictionary = MergeDictionary(tuple(entries))
        encodings = encode_corpus(streams, dictionary, workers=workers)
        usage = usage_counts(encodings, dictionary)
        expansions = {
            dictionary.token_id(index): atoms for index, atoms in enumerate(entries)
        }
        entries = [atoms for atoms, count in zip(entries, usage) if count > 0]
        known = set(entries)
        added = 0
        for atoms in _merge_counts(encodings, expansions, pool):
            if atoms in known:
                continue
            entries.append(atoms)
            known.add(atoms)
            added += 1
            if added >= block or len(entries) >= budget:
                break
        if added == 0:
            break
    if entries:
        dictionary = MergeDictionary(tuple(entries))
        encodings = encode_corpus(streams, dictionary, workers=workers)
        usage = usage_counts(encodings, dictionary)
        entries = [atoms for atoms, count in zip(entries, usage) if count > 0]
    return MergeDictionary(tuple(entries))
