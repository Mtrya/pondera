"""Fit and measure merge dictionaries on declared trace corpora.

Full presentations supply candidate substrings. They are distinct sampled
roots, rather than alternate full views of every random/zero row. Measurement
includes every selected presentation; vocabulary holdout is optional.
"""

import argparse
import json
import random
import resource
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pondera.chess.merge import (
    MergeDictionary,
    MergeEncoder,
    count_candidates,
    decode,
    select_dictionary,
    usage_counts,
)
from pondera.chess.vocabulary import CHESS_TOKENIZER_VERSION, CHESS_VOCAB_SIZE, LINES
from pondera.data.dataset import load_dictionary, load_manifest, parquet_sources
from pondera.data.text import trace_atoms
from pondera.data.trace import TRACE_GRAMMAR_VERSION

DEFAULT_BUDGETS = (100, 500, 2000, 8000)


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


def load_rows(release, *, max_roots=None):
    """Project only profile columns, selecting root groups before token IO."""
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
    dictionary = load_dictionary(release) if release.is_dir() else MergeDictionary()
    rows = []
    for path, expected, split in sources:
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
        cache = "trace_token_ids" in file.schema_arrow.names
        columns.append("trace_token_ids" if cache else "trace_text")
        for batch in file.iter_batches(
            columns=columns, batch_size=64, use_threads=False
        ):
            for row in batch.to_pylist():
                position = " ".join(row["fen"].split()[:4])
                if selected is not None and position not in selected:
                    continue
                label = row.get("split", split)
                if label not in {"train", "val"}:
                    raise ValueError(f"Unknown dataset split: {label}")
                atoms = (
                    decode(row["trace_token_ids"], dictionary).astype(np.int32)
                    if cache
                    else trace_atoms(row["trace_text"])
                )
                rows.append(
                    ProfileRow(
                        str(release),
                        row.get("root_id") or f"{row['fen']}|{row['repetition']}",
                        position,
                        label,
                        row["presentation"],
                        row["teacher_width"],
                        row["teacher_depth"],
                        atoms,
                    )
                )
    return rows


def load_full_rows(release):
    """Full rows, retaining split, identity, and the declared cache encoding."""
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


def _measure(rows, dictionary):
    encoder = MergeEncoder(dictionary)
    total = _empty()
    usage = np.zeros(len(dictionary), dtype=np.int64)
    slices = {
        name: {}
        for name in ("presentation", "search", "length", "release_split", "release")
    }
    lengths = []
    for row in rows:
        encoded = encoder.encode(row.atoms)
        atoms, tokens = len(row.atoms), len(encoded)
        _add(total, atoms, tokens)
        usage += usage_counts((encoded,), dictionary)
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
    budgets,
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
    on_tier=None,
):
    """Count full rows, select dictionaries, and stream corpus measurements."""
    started = time.monotonic()
    releases = [Path(release).resolve() for release in releases]
    budgets = list(budgets)
    if not releases or len(set(releases)) != len(releases):
        raise ValueError("Provide distinct input releases")
    if (
        not budgets
        or any(budget < 0 for budget in budgets)
        or len(set(budgets)) != len(budgets)
    ):
        raise ValueError("Provide distinct nonnegative budgets")
    if not 0 <= holdout_fraction < 1:
        raise ValueError("holdout fraction must lie in [0, 1)")
    if selection_stride < 1 or fit_split not in {"all", "train", "val"}:
        raise ValueError("Invalid fitting split or selection stride")
    loaded = [
        row for release in releases for row in load_rows(release, max_roots=max_roots)
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
    if not fitting:
        raise ValueError("The fitting corpus has no full-presentation rows")
    measurement = [row for row in loaded if row.position not in heldout_positions]
    heldout = [row for row in loaded if row.position in heldout_positions]
    positions = sorted({row.position for row in fitting})
    selection_positions = set(positions[::selection_stride])
    selection = [row.atoms for row in fitting if row.position in selection_positions]
    load_seconds = time.monotonic() - started
    count_start = time.monotonic()
    candidates = count_candidates(
        (row.atoms for row in fitting),
        min_count=min_count,
        per_length_limit=per_length_limit,
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
                "rows": len(selection),
                "atoms": sum(map(len, selection)),
            },
        },
        "candidates": {
            "min_count": min_count,
            "per_length_limit": per_length_limit,
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
    for budget in budgets:
        round_block = block if block is not None else max(8, budget // 32)
        selection_start = time.monotonic()
        dictionary = select_dictionary(
            selection,
            candidates,
            budget,
            block=round_block,
            max_rounds=max_rounds,
        )
        select_seconds = time.monotonic() - selection_start
        measure_start = time.monotonic()
        measured, usage = _measure(measurement, dictionary)
        unseen, _ = _measure(heldout, dictionary)
        dictionaries[budget] = dictionary
        document["tiers"].append(
            {
                "budget": budget,
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
    document["timing"]["peak_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
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
        "| budget | entries | encoded tokens | corpus ratio | active entries | held-out ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in document["tiers"]:
        measured = tier["measurement"]
        lines.append(
            f"| {tier['budget']:,} | {tier['entries']:,} | {measured['tokens']:,} | "
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
            f"### Budget {tier['budget']}",
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
    for budget, dictionary in dictionaries.items():
        dictionary.write(output / f"dictionary-{budget}.tsv")
    (output / "curve.json").write_text(json.dumps(document, indent=2) + "\n")
    (output / "profile.md").write_text(render_report(document))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget", action="append", type=int)
    parser.add_argument("--fit-split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--holdout-fraction", type=float, default=0.0)
    parser.add_argument("--max-roots", type=int)
    parser.add_argument("--min-count", type=int, default=8)
    parser.add_argument("--per-length-limit", type=int, default=20_000)
    parser.add_argument("--selection-stride", type=int, default=1)
    parser.add_argument("--block", type=int, help="default max(8, budget // 32)")
    parser.add_argument("--max-rounds", type=int, default=100_000)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    def report(tier):
        print(
            f"budget {tier['budget']:,}: {tier['entries']:,} entries; "
            f"corpus {_percent(tier['measurement']['ratio'])}; "
            f"held out {_percent(tier['heldout']['ratio'])}",
            flush=True,
        )

    document, dictionaries = build_profile(
        args.release,
        args.budget or DEFAULT_BUDGETS,
        fit_split=args.fit_split,
        holdout_fraction=args.holdout_fraction,
        max_roots=args.max_roots,
        min_count=args.min_count,
        per_length_limit=args.per_length_limit,
        selection_stride=args.selection_stride,
        block=args.block,
        max_rounds=args.max_rounds,
        top=args.top,
        on_tier=report,
    )
    write_artifacts(args.output, document, dictionaries)
    print(f"artifacts: {args.output}")


if __name__ == "__main__":
    main()
