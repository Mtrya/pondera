"""Fit and measure merge dictionaries on declared trace corpora.

Full presentations supply candidate substrings. They are distinct sampled
roots, rather than alternate full views of every random/zero row. Measurement
includes every selected presentation; vocabulary holdout is optional.
"""

import json
import multiprocessing
import random
import resource
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pondera.chess.merge import (
    count_candidates,
    select_dictionary,
    usage_counts,
)
from pondera.chess.vocabulary import (
    CHESS_TOKENIZER_VERSION,
    CHESS_VOCAB_SIZE,
    LINES,
)
from pondera.data.dataset import load_manifest, parquet_sources
from pondera.data.text import (
    MERGE_BOUNDARIES,
    encode_traces,
    merge_segments,
    trace_atoms,
)
from pondera.data.trace import TRACE_GRAMMAR_VERSION


@dataclass(frozen=True)
class ProfileRow:
    release: str
    root_id: str
    position: str
    split: str
    presentation: str
    width: int
    depth: int
    atoms: np.ndarray


def token_name(token):
    return LINES[token]


def is_held_out(root_id, fraction):
    return random.Random(root_id).random() < fraction


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )


def _load_shard(task):
    """Parse one shard's profile rows from authoritative text, in order."""
    path, expected, split, release_name, selected = task
    file = pq.ParquetFile(path)
    if file.metadata.num_rows != expected:
        raise ValueError(f"Shard row count differs from its release: {path}")
    columns = [
        "fen",
        "repetition",
        "presentation",
        "teacher_width",
        "teacher_depth",
    ]
    columns.extend(
        name for name in ("root_id", "split") if name in file.schema_arrow.names
    )
    columns.append("trace_text")
    rows = []
    for batch in file.iter_batches(columns=columns, batch_size=64, use_threads=False):
        for row in batch.to_pylist():
            position = " ".join(row["fen"].split()[:4])
            if selected is not None and position not in selected:
                continue
            label = row.get("split", split)
            if label not in {"train", "val"}:
                raise ValueError(f"Unknown dataset split: {label}")
            rows.append(
                ProfileRow(
                    release_name,
                    row.get("root_id") or f"{row['fen']}|{row['repetition']}",
                    position,
                    label,
                    row["presentation"],
                    row["teacher_width"],
                    row["teacher_depth"],
                    trace_atoms(row["trace_text"]),
                )
            )
    return rows


def load_rows(release, *, max_roots=None, workers=1):
    """Project only profile columns, selecting root groups before token IO.

    With ``workers`` above one, spawned worker processes parse shards in
    parallel — spawned, not forked, so no Parquet IO state crosses a fork.
    Rows keep shard order either way.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    release = Path(release)
    manifest = load_manifest(release / "release.json") if release.is_dir() else {}
    if manifest:
        for field, expected in (
            ("grammar_version", TRACE_GRAMMAR_VERSION),
            ("tokenizer_version", CHESS_TOKENIZER_VERSION),
        ):
            if manifest.get(field) != expected:
                raise ValueError(f"Unsupported {field} in {release}")
    sources = parquet_sources(release)
    selected = None
    if max_roots is not None:
        if max_roots < 1:
            raise ValueError("max_roots must be positive")
        positions = set()
        for path, _, _ in sources:
            for batch in pq.ParquetFile(path).iter_batches(
                columns=["fen"], batch_size=1024, use_threads=False
            ):
                positions.update(
                    " ".join(fen.split()[:4]) for fen in batch.column(0).to_pylist()
                )
        selected = set(
            sorted(
                positions, key=lambda key: (random.Random(f"640:{key}").random(), key)
            )[:max_roots]
        )
    tasks = [
        (path, expected, split, str(release), selected)
        for path, expected, split in sources
    ]
    if workers > 1 and len(tasks) > 1:
        context = multiprocessing.get_context("spawn")
        with context.Pool(workers) as pool:
            groups = pool.map(_load_shard, tasks)
    else:
        groups = [_load_shard(task) for task in tasks]
    return [row for group in groups for row in group]


def load_full_rows(release):
    """Full rows from authoritative text, retaining split and identity."""
    return [row for row in load_rows(release) if row.presentation == "full"]


def _empty():
    return {"rows": 0, "atoms": 0, "tokens": 0}


def _add(total, atoms, tokens):
    total["rows"] += 1
    total["atoms"] += atoms
    total["tokens"] += tokens


def _ratio(total):
    return {
        **total,
        "ratio": total["tokens"] / total["atoms"] if total["atoms"] else None,
    }


def _summary(rows):
    used = np.zeros(CHESS_VOCAB_SIZE, dtype=bool)
    for row in rows:
        used[row.atoms] = True
    return {
        "rows": len(rows),
        "roots": len({row.position for row in rows}),
        "atoms": sum(len(row.atoms) for row in rows),
        "distinct_atoms": int(used.sum()),
        "release_splits": dict(Counter(row.split for row in rows)),
        "presentations": dict(Counter(row.presentation for row in rows)),
    }


def _measure(rows, dictionary, workers=1):
    encodings = encode_traces((row.atoms for row in rows), dictionary, workers=workers)
    usage = usage_counts(encodings, dictionary)
    total = _empty()
    slices = {
        name: {}
        for name in ("presentation", "search", "length", "release_split", "release")
    }
    lengths = []
    for row, encoded in zip(rows, encodings):
        atoms, tokens = len(row.atoms), len(encoded)
        _add(total, atoms, tokens)
        lengths.append(tokens)
        length = (
            "<1k"
            if atoms < 1024
            else "1k-8k"
            if atoms < 8192
            else "8k-32k"
            if atoms < 32768
            else "32k+"
        )
        keys = {
            "presentation": row.presentation,
            "search": f"W{row.width}/D{row.depth}",
            "length": length,
            "release_split": row.split,
            "release": row.release,
        }
        for kind, key in keys.items():
            _add(slices[kind].setdefault(key, _empty()), atoms, tokens)
    result = {
        **_ratio(total),
        "used": int((usage > 0).sum()),
        "utilization": int((usage > 0).sum()) / len(dictionary)
        if len(dictionary)
        else None,
        "encoded_length": (
            dict(
                zip(
                    ("p50", "p95", "max"),
                    map(float, np.quantile(lengths, (0.5, 0.95, 1))),
                )
            )
            if lengths
            else None
        ),
        "slices": {
            kind: {key: _ratio(value) for key, value in groups.items()}
            for kind, groups in slices.items()
        },
    }
    return result, usage


def _top(dictionary, usage, count):
    order = sorted(
        range(len(dictionary)),
        key=lambda index: (
            -int(usage[index]) * (len(dictionary.entries[index]) - 1),
            -int(usage[index]),
            dictionary.entries[index],
        ),
    )
    return [
        {
            "id": dictionary.token_id(index),
            "atoms": list(dictionary.entries[index]),
            "name": " ".join(token_name(atom) for atom in dictionary.entries[index]),
            "usage": int(usage[index]),
            "savings": int(usage[index]) * (len(dictionary.entries[index]) - 1),
        }
        for index in order[:count]
    ]


def build_profile(
    releases,
    vocab_sizes,
    *,
    holdout_fraction=0.0,
    fit_split="all",
    max_roots=None,
    min_count=8,
    per_length_limit=20_000,
    selection_stride=1,
    block=None,
    max_rounds=100_000,
    top=20,
    workers=1,
    on_tier=None,
):
    """Count full rows, select dictionaries, and stream corpus measurements.

    ``workers`` parallelizes loading, candidate counting, selection encoding,
    and measurement across processes without changing any result.
    """
    started = time.monotonic()
    releases = [Path(release).resolve() for release in releases]
    vocab_sizes = list(vocab_sizes)
    if not releases or len(set(releases)) != len(releases):
        raise ValueError("Provide distinct input releases")
    if (
        not vocab_sizes
        or any(size < CHESS_VOCAB_SIZE for size in vocab_sizes)
        or len(set(vocab_sizes)) != len(vocab_sizes)
    ):
        raise ValueError(
            "Provide distinct vocabulary sizes no smaller than the base vocabulary"
        )
    if not 0 <= holdout_fraction < 1:
        raise ValueError("holdout fraction must lie in [0, 1)")
    if selection_stride < 1 or fit_split not in {"all", "train", "val"}:
        raise ValueError("Invalid fitting split or selection stride")
    if workers < 1:
        raise ValueError("workers must be positive")
    loaded = [
        row
        for release in releases
        for row in load_rows(release, max_roots=max_roots, workers=workers)
    ]
    heldout_positions = {
        row.position for row in loaded if is_held_out(row.position, holdout_fraction)
    }
    if fit_split != "all":
        heldout_positions.update(
            row.position for row in loaded if row.split != fit_split
        )
    fitting = [
        row
        for row in loaded
        if row.presentation == "full" and row.position not in heldout_positions
    ]
    if not fitting and any(size > CHESS_VOCAB_SIZE for size in vocab_sizes):
        raise ValueError("The fitting corpus has no full-presentation rows")
    measurement = [row for row in loaded if row.position not in heldout_positions]
    heldout = [row for row in loaded if row.position in heldout_positions]
    positions = sorted({row.position for row in fitting})
    selection_positions = set(positions[::selection_stride])
    selection_rows = [row for row in fitting if row.position in selection_positions]
    selection = [
        segment for row in selection_rows for segment in merge_segments(row.atoms)
    ]
    load_seconds = time.monotonic() - started
    print(
        f"loaded {len(loaded)} rows ({len(fitting)} fitting) in {load_seconds:.0f}s",
        flush=True,
    )
    count_start = time.monotonic()
    candidates = (
        count_candidates(
            (segment for row in fitting for segment in merge_segments(row.atoms)),
            min_count=min_count,
            per_length_limit=per_length_limit,
            workers=workers,
        )
        if any(size > CHESS_VOCAB_SIZE for size in vocab_sizes)
        else []
    )
    print(
        f"counted {len(candidates)} candidates in "
        f"{time.monotonic() - count_start:.0f}s",
        flush=True,
    )
    document = {
        "corpus": {
            "releases": [str(path) for path in releases],
            "max_roots_per_release": max_roots,
            "fit_split": fit_split,
            "holdout_fraction": holdout_fraction,
            "loaded": _summary(loaded),
            "fitting": _summary(fitting),
            "measurement": _summary(measurement),
            "heldout": _summary(heldout),
            "selection": {
                "stride": selection_stride,
                "roots": len(selection_positions),
                "rows": len(selection_rows),
                "atoms": sum(map(len, selection)),
            },
        },
        "candidates": {
            "min_count": min_count,
            "per_length_limit": per_length_limit,
            "boundaries": MERGE_BOUNDARIES,
            "counted": len(candidates),
            "retained_by_length": dict(
                Counter(len(candidate.atoms) for candidate in candidates)
            ),
        },
        "timing": {
            "load_seconds": load_seconds,
            "count_seconds": time.monotonic() - count_start,
        },
        "tiers": [],
    }
    dictionaries = {}
    for size in vocab_sizes:
        composites = size - CHESS_VOCAB_SIZE
        round_block = block if block is not None else max(8, composites // 32)
        selection_start = time.monotonic()
        dictionary = select_dictionary(
            selection,
            candidates,
            composites,
            block=round_block,
            max_rounds=max_rounds,
            workers=workers,
        )
        select_seconds = time.monotonic() - selection_start
        print(
            f"selected {len(dictionary)} entries for vocab {size} in "
            f"{select_seconds:.0f}s",
            flush=True,
        )
        measure_start = time.monotonic()
        measured, usage = _measure(measurement, dictionary, workers=workers)
        unseen, _ = _measure(heldout, dictionary, workers=workers)
        print(f"measured in {time.monotonic() - measure_start:.0f}s", flush=True)
        dictionaries[size] = dictionary
        document["tiers"].append(
            {
                "vocab_size": size,
                "block": round_block,
                "entries": len(dictionary),
                "measurement": measured,
                "heldout": unseen,
                "top": _top(dictionary, usage, top),
                "select_seconds": select_seconds,
                "measure_seconds": time.monotonic() - measure_start,
            }
        )
        if on_tier is not None:
            on_tier(document["tiers"][-1])
    document["timing"]["total_seconds"] = time.monotonic() - started
    document["timing"]["peak_rss_bytes"] = peak_rss_bytes()
    return document, dictionaries


def _percent(value):
    return "n/a" if value is None else f"{100 * value:.2f}%"


def render_report(document, title="Merge dictionary profile"):
    """Describe the measured corpus, optional partition, and compression curve."""
    corpus = document["corpus"]
    lines = [
        f"# {title}",
        "",
        f"Candidate counting uses {corpus['fitting']['rows']:,} full rows; selection uses "
        f"{corpus['selection']['rows']:,} rows. Measurement covers "
        f"{corpus['measurement']['rows']:,} rows across their stored presentations. "
        "Each root has its selected presentation; missing full views are not regenerated.",
        "",
        f"Fitting split: {corpus['fit_split']}; optional root holdout: "
        f"{corpus['holdout_fraction']:.3f}. Original dataset split labels are preserved.",
        "",
        "| vocab size | entries | encoded tokens | corpus ratio | active entries | held-out ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in document["tiers"]:
        measured = tier["measurement"]
        lines.append(
            f"| {tier['vocab_size']:,} | {tier['entries']:,} | {measured['tokens']:,} | "
            f"{_percent(measured['ratio'])} | {measured['used']:,} | "
            f"{_percent(tier['heldout']['ratio'])} |"
        )
    lines += [
        "",
        "Ratio is encoded tokens / base atoms. n/a means no measurement, not zero compression.",
        "",
        "## Resources",
        "",
        f"Total: {document['timing']['total_seconds']:.3f} s; "
        f"process peak RSS: {document['timing']['peak_rss_bytes']:,} bytes.",
        "",
        "## Top composites",
    ]
    for tier in document["tiers"]:
        lines += [
            "",
            f"### Vocabulary size {tier['vocab_size']}",
            "",
            "| id | uses | atom savings | expansion |",
            "|---:|---:|---:|---|",
        ]
        lines.extend(
            f"| {entry['id']} | {entry['usage']:,} | {entry['savings']:,} | {entry['name']} |"
            for entry in tier["top"]
        )
    return "\n".join(lines) + "\n"


def write_artifacts(output, document, dictionaries):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    document = {
        **document,
        "tiers": [
            {**tier, "merge_entries": dictionaries[tier["vocab_size"]].entries}
            for tier in document["tiers"]
        ],
    }
    (output / "curve.json").write_text(json.dumps(document, indent=2) + "\n")
    (output / "profile.md").write_text(render_report(document))
