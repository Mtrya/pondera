"""Merge-vocabulary profile over grammar-v2 trace releases.

A profile reads the full-presentation rows of one or more releases, counts
atom substrings on the training roots, selects merge dictionaries at the
requested budgets, and writes the compression curve, the top composites of
every budget, and the dictionary files.

Only full rows are used. The zero and random presentations are truncations of
the same recorded search, so every substring they contain already occurs in
the full row; counting them would weight shared prefixes without adding
substring structure.

Roots are split by a deterministic draw from their id: the held-out tenth
never enters counting or selection and only measures what the curve reports.
Selection itself runs on every fourth training root, which keeps the encoder
passes affordable without touching held-out data.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from pondera.chess.merge import (
    MergeDictionary,
    count_candidates,
    encode_corpus,
    select_dictionary,
    usage_counts,
)
from pondera.chess.vocabulary import (
    CHESS_TOKENIZER_VERSION,
    CHESS_VOCAB_SIZE,
    LINES,
)
from pondera.data.dataset import load_manifest, load_trace_arrays, read_root_rows
from pondera.data.trace import TRACE_GRAMMAR_VERSION

FULL_PRESENTATION = "full"
DEFAULT_BUDGETS = (100, 500, 2000, 8000)


def token_name(token: int) -> str:
    return LINES[token] if 0 <= token < len(LINES) else f"#{token}"


def load_full_rows(release):
    """The full-presentation rows of one release: (root_id, atoms) pairs."""
    release = Path(release)
    manifest = load_manifest(release / "release.json")
    if not manifest:
        raise FileNotFoundError(release / "release.json")
    expected = {
        "grammar_version": TRACE_GRAMMAR_VERSION,
        "tokenizer_version": CHESS_TOKENIZER_VERSION,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"{release} is not a supported release: {field} is not {value}"
            )
    rows = []
    for shard in manifest["shards"]:
        directory = release / shard["path"]
        tokens = load_trace_arrays(directory, ("trace_token_ids",))["trace_token_ids"]
        records = read_root_rows(directory)
        for index, record in enumerate(records):
            if record["presentation"] != FULL_PRESENTATION:
                continue
            atoms = np.asarray(tokens[index], dtype=np.int64)
            rows.append((record["root_id"], atoms))
    return rows


def is_held_out(root_id: str, fraction: float) -> bool:
    """Stable root draw: the same id always lands on the same side."""
    return random.Random(root_id).random() < fraction


def split_roots(rows, fraction):
    """Group rows by root id, then split whole roots into train and held-out."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError("holdout fraction must lie in [0, 1)")
    groups = {}
    for release, root_id, atoms in rows:
        groups.setdefault(root_id, []).append((release, atoms))
    train, heldout = {}, {}
    for root_id, entries in groups.items():
        (heldout if is_held_out(root_id, fraction) else train)[root_id] = entries
    return train, heldout


def _flatten(groups):
    streams, owners = [], []
    for root_id in sorted(groups):
        for release, atoms in groups[root_id]:
            streams.append(atoms)
            owners.append(release)
    return streams, owners


def _sample(groups, stride: int):
    """Every ``stride``-th training root, in sorted id order."""
    if stride < 1:
        raise ValueError("selection stride must be positive")
    return [
        atoms
        for index, root_id in enumerate(sorted(groups))
        if index % stride == 0
        for _, atoms in groups[root_id]
    ]


def _atoms_used(streams) -> int:
    if not streams:
        return 0
    counts = np.bincount(np.concatenate(streams), minlength=CHESS_VOCAB_SIZE)
    return int((counts > 0).sum())


def _side(streams, encodings, usage, entries: int) -> dict:
    atoms = sum(len(stream) for stream in streams)
    tokens = sum(len(encoded) for encoded in encodings)
    used = int((usage > 0).sum())
    return {
        "atoms": atoms,
        "tokens": tokens,
        "ratio": tokens / atoms if atoms else 0.0,
        "used": used,
        "utilization": used / entries if entries else 0.0,
    }


def _release_totals(owners, streams, encodings) -> dict:
    totals = {}
    for owner, atoms, encoded in zip(owners, streams, encodings):
        entry = totals.setdefault(str(owner), {"atoms": 0, "tokens": 0})
        entry["atoms"] += len(atoms)
        entry["tokens"] += len(encoded)
    return totals


def _per_release(paths, train_totals, heldout_totals) -> list:
    result = []
    for path in sorted({str(path) for path in paths}):
        entry = {"path": path}
        for side, totals in (("train", train_totals), ("heldout", heldout_totals)):
            atoms = totals.get(path, {"atoms": 0, "tokens": 0})["atoms"]
            tokens = totals.get(path, {"atoms": 0, "tokens": 0})["tokens"]
            entry[side] = {
                "atoms": atoms,
                "tokens": tokens,
                "ratio": tokens / atoms if atoms else 0.0,
            }
        result.append(entry)
    return result


def _top(dictionary: MergeDictionary, usage, count: int) -> list:
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
            "train_usage": int(usage[index]),
            "savings": int(usage[index]) * (len(dictionary.entries[index]) - 1),
        }
        for index in order[:count]
    ]


def build_profile(
    releases,
    budgets,
    *,
    holdout_fraction: float = 0.1,
    min_count: int = 8,
    per_length_limit: int = 20_000,
    selection_stride: int = 4,
    block=None,
    max_rounds: int = 100_000,
    top: int = 20,
):
    """Count, select, and measure; returns the curve document and dictionaries."""
    releases = [Path(release) for release in releases]
    if not releases:
        raise ValueError("A profile needs at least one release")
    if len(set(releases)) != len(releases):
        raise ValueError("A release may appear only once")
    loaded = [(release, load_full_rows(release)) for release in releases]
    rows = [
        (release, root_id, atoms)
        for release, pairs in loaded
        for root_id, atoms in pairs
    ]
    train_groups, heldout_groups = split_roots(rows, holdout_fraction)
    train_streams, train_owners = _flatten(train_groups)
    heldout_streams, heldout_owners = _flatten(heldout_groups)
    if not train_streams:
        raise ValueError("The held-out split removed every training root")

    selection_streams = _sample(train_groups, selection_stride)
    candidates = count_candidates(
        train_streams, min_count=min_count, per_length_limit=per_length_limit
    )
    document = {
        "corpus": {
            "holdout_fraction": holdout_fraction,
            "releases": [
                {
                    "path": str(release),
                    "roots": len({root_id for root_id, _ in pairs}),
                    "rows": len(pairs),
                    "atoms": sum(len(atoms) for _, atoms in pairs),
                }
                for release, pairs in loaded
            ],
            "train": {
                "roots": len(train_groups),
                "rows": len(train_streams),
                "atoms": sum(len(stream) for stream in train_streams),
                "distinct_atoms": _atoms_used(train_streams),
            },
            "heldout": {
                "roots": len(heldout_groups),
                "rows": len(heldout_streams),
                "atoms": sum(len(stream) for stream in heldout_streams),
                "distinct_atoms": _atoms_used(heldout_streams),
            },
            "selection": {
                "stride": selection_stride,
                "roots": len(range(0, len(train_groups), selection_stride)),
                "rows": len(selection_streams),
                "atoms": sum(len(stream) for stream in selection_streams),
            },
        },
        "candidates": {
            "min_count": min_count,
            "per_length_limit": per_length_limit,
            "counted": len(candidates),
        },
        "tiers": [],
    }
    dictionaries = {}
    for budget in budgets:
        round_block = block if block is not None else max(8, budget // 32)
        dictionary = select_dictionary(
            selection_streams,
            candidates,
            budget,
            block=round_block,
            max_rounds=max_rounds,
        )
        dictionaries[budget] = dictionary
        train_encodings = encode_corpus(train_streams, dictionary)
        heldout_encodings = encode_corpus(heldout_streams, dictionary)
        train_usage = usage_counts(train_encodings, dictionary)
        heldout_usage = usage_counts(heldout_encodings, dictionary)
        document["tiers"].append(
            {
                "budget": budget,
                "block": round_block,
                "entries": len(dictionary),
                "train": _side(
                    train_streams, train_encodings, train_usage, len(dictionary)
                ),
                "heldout": _side(
                    heldout_streams, heldout_encodings, heldout_usage, len(dictionary)
                ),
                "per_release": _per_release(
                    releases,
                    _release_totals(train_owners, train_streams, train_encodings),
                    _release_totals(heldout_owners, heldout_streams, heldout_encodings),
                ),
                "top": _top(dictionary, train_usage, top),
            }
        )
    return document, dictionaries


def _percent(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}%"


def render_report(document, title: str = "Merge dictionary profile") -> str:
    """Markdown summary of one curve document."""
    corpus = document["corpus"]
    candidates = document["candidates"]
    train, heldout = corpus["train"], corpus["heldout"]
    lines = [
        f"# {title}",
        "",
        (
            "The corpus is the full-presentation rows of the releases below. The zero "
            "and random presentations truncate the same search, so they add "
            "no substring structure of their own. Counting saw every training root; "
            f"selection tuned on every {corpus['selection']['stride']}th training root "
            f"({corpus['selection']['roots']:,} roots); the held-out "
            f"{_percent(corpus['holdout_fraction'], 0)} of roots (a stable draw on the "
            "root id) measures the curve."
        ),
        "",
        "| release | roots | full rows | atoms |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| `{release['path']}` | {release['roots']:,} | {release['rows']:,} | "
        f"{release['atoms']:,} |"
        for release in corpus["releases"]
    ]
    lines += [
        "",
        f"Train: {train['roots']:,} roots, {train['rows']:,} rows, "
        f"{train['atoms']:,} atoms ({train['distinct_atoms']:,} of "
        f"{CHESS_VOCAB_SIZE:,} atoms used).",
        "",
        f"Held out: {heldout['roots']:,} roots, {heldout['rows']:,} rows, "
        f"{heldout['atoms']:,} atoms ({heldout['distinct_atoms']:,} atoms used).",
        "",
        f"Candidates: {candidates['counted']:,} substrings of "
        f"{candidates['min_count']}+ occurrences, top "
        f"{candidates['per_length_limit']:,} per length are admissible merges.",
        "",
        "## Compression curve",
        "",
        "| budget | entries | train tokens | train ratio | held-out tokens | "
        "held-out ratio | held-out use |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in document["tiers"]:
        lines.append(
            f"| {tier['budget']:,} | {tier['entries']:,} | "
            f"{tier['train']['tokens']:,} | {_percent(tier['train']['ratio'])} | "
            f"{tier['heldout']['tokens']:,} | {_percent(tier['heldout']['ratio'])} | "
            f"{_percent(tier['heldout']['utilization'], 1)} |"
        )
    lines += [
        "",
        (
            "Ratio = encoded tokens / atom tokens; held-out use = composites the "
            "held-out rows actually chose / dictionary size. Selection keeps only "
            "composites its encoder chose, so training use is near 100%; the held-out "
            "column is what an unseen root pays for the vocabulary."
        ),
        "",
        "## Held-out ratio by release",
        "",
        "| release | "
        + " | ".join(f"{tier['budget']:,}" for tier in document["tiers"])
        + " |",
        "|---|" + "---:|" * len(document["tiers"]),
    ]
    paths = sorted(
        {entry["path"] for tier in document["tiers"] for entry in tier["per_release"]}
    )
    for path in paths:
        cells = []
        for tier in document["tiers"]:
            entry = next(item for item in tier["per_release"] if item["path"] == path)
            cells.append(_percent(entry["heldout"]["ratio"]))
        lines.append(f"| `{path}` | " + " | ".join(cells) + " |")
    lines += ["", "## Top merges"]
    for tier in document["tiers"]:
        lines += [
            "",
            f"### {tier['budget']:,} merges",
            "",
            "| id | uses | savings | expansion |",
            "|---:|---:|---:|---|",
        ]
        lines += [
            f"| {entry['id']} | {entry['train_usage']:,} | {entry['savings']:,} | "
            f"`{entry['name']}` |"
            for entry in tier["top"]
        ]
    lines.append("")
    return "\n".join(lines)


def write_artifacts(output, document, dictionaries) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for budget, dictionary in dictionaries.items():
        dictionary.write(output / f"dictionary-{budget}.tsv")
    (output / "curve.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    (output / "profile.md").write_text(render_report(document), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        action="append",
        required=True,
        type=Path,
        help="release directory; repeat for every corpus",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--budget",
        action="append",
        type=int,
        help=f"merge budget; repeatable, default {list(DEFAULT_BUDGETS)}",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--min-count", type=int, default=8)
    parser.add_argument("--per-length-limit", type=int, default=20_000)
    parser.add_argument("--selection-stride", type=int, default=4)
    parser.add_argument(
        "--block",
        type=int,
        help="pairs admitted per encoder round; default max(8, budget // 32)",
    )
    parser.add_argument("--max-rounds", type=int, default=100_000)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    budgets = args.budget or list(DEFAULT_BUDGETS)
    document, dictionaries = build_profile(
        args.release,
        budgets,
        holdout_fraction=args.holdout_fraction,
        min_count=args.min_count,
        per_length_limit=args.per_length_limit,
        selection_stride=args.selection_stride,
        block=args.block,
        max_rounds=args.max_rounds,
        top=args.top,
    )
    write_artifacts(args.output, document, dictionaries)
    for tier in document["tiers"]:
        print(
            f"budget {tier['budget']:>6,}: {tier['entries']:>6,} entries, "
            f"train {_percent(tier['train']['ratio'])}, "
            f"held-out {_percent(tier['heldout']['ratio'])} "
            f"({_percent(tier['heldout']['utilization'], 1)} used)"
        )
    print(f"artifacts: {args.output}")


if __name__ == "__main__":
    main()
