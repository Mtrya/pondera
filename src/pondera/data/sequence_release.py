"""Sharded, resumable grammar-v2 sequence releases.

A release row is one presentation of one root. By default all four
presentations are written; a recipe may sample one presentation per root.
Every written presentation of a root shares the complete search's label
suffix, so they predict the same final move, evaluation, and variation. The
builder is resumable by root: a published shard is verified and reused, and
the release manifest is updated after each shard.
"""

import json
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pondera.chess.codec import encode_board
from pondera.chess.vocabulary import CHESS_TOKENIZER_VERSION, PAD
from pondera.data.sequence_format import SEQUENCE_FORMAT_VERSION
from pondera.data.sequence_validation import verify_shard_files
from pondera.data.sidecar import (
    SIDECAR_SCHEMA,
    SIDECAR_SCHEMA_WIDE,
    load_sidecar,
)
from pondera.data.storage import content_sha256, load_manifest, sha256, write_manifest
from pondera.data.trace import ROLE_COUNT, TRACE_GRAMMAR_VERSION
from pondera.data.trace_config import DEFAULT_TEACHER_PATH, sample_presentation
from pondera.data.trace_profile import RootProfile
from pondera.data.trace_synthesis import PRESENTATIONS, synthesize_root

RELEASE_ONLY_METADATA = frozenset(
    {"root_selection", "root_selections", "roots", "root_sets", "sidecars"}
)
ROOT_ROW_FIELDS = ("split", "fen", "repetition", "turn", "position_key", "source")
CHAMPION_FIELDS = ("round", "depth", "move", "raw", "wdl", "mate", "value_token", "pv")


def _write_jsonl(path, rows):
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _array(directory, name, rows, width, dtype, padding):
    array = np.lib.format.open_memmap(
        directory / f"{name}.npy", mode="w+", dtype=dtype, shape=(len(rows), width)
    )
    array[:] = padding
    for index, values in enumerate(rows):
        array[index, : len(values)] = values
    array.flush()
    return array


def _one_dimensional(directory, name, values, dtype):
    array = np.lib.format.open_memmap(
        directory / f"{name}.npy", mode="w+", dtype=dtype, shape=(len(values),)
    )
    array[:] = values
    array.flush()
    return array


def _root_row(root, synthesized, presentation):
    row = synthesized.rows[presentation]
    payload = {
        "root_id": root["root_id"],
        "presentation": presentation,
        "teacher_width": synthesized.stats["teacher_width"],
        "depth": synthesized.stats["depth"],
        "rounds": row.rounds,
        "thinking_tokens": row.thinking_tokens,
        "final_move": row.final_move,
        "final_value_bin": row.final_value_bin,
        "mate_distance": row.mate_distance,
        "claim_draw": row.claim_draw,
    }
    for name in ROOT_ROW_FIELDS:
        if name in root:
            payload[name] = root[name]
    return payload


def _champion_row(root, synthesized):
    rounds = []
    for audit in synthesized.round_audits:
        entry = {
            "round": audit.round,
            "depth": audit.depth,
            "move": audit.move,
            "raw": audit.raw,
            "wdl": None if audit.wdl is None else list(audit.wdl),
            "mate": audit.mate,
            "value_token": audit.value_token,
            "pv": list(audit.pv),
            "refutations": audit.refutations,
            "candidates": [
                {
                    "move": candidate.move,
                    "raw": candidate.raw,
                    "wdl": None if candidate.wdl is None else list(candidate.wdl),
                    "mate": candidate.mate,
                    "value_token": candidate.value_token,
                    "order": candidate.order,
                    "draw": candidate.draw,
                }
                for candidate in audit.candidates
            ],
        }
        rounds.append(entry)
    return {
        "root_id": root["root_id"],
        "rounds": rounds,
        "stats": synthesized.stats,
    }


def _write_shard(directory, records, metadata, presentations):
    boards, traces, roles, lengths = [], [], [], []
    no_tool, tool, best, basis, values, counts = [], [], [], [], [], []
    root_rows = []
    champion_rows = []
    for synthesized in records:
        root = synthesized.root
        board = encode_board(root["fen"], int(root["repetition"]))
        for presentation in presentations[root["root_id"]]:
            row = synthesized.rows[presentation]
            boards.append(board)
            traces.append(row.tokens)
            roles.append(row.roles)
            lengths.append(len(row.tokens))
            no_tool.append([boundary.no_tool_end for boundary in row.boundaries])
            tool.append([boundary.tool_end for boundary in row.boundaries])
            best.append([boundary.best for boundary in row.boundaries])
            basis.append([boundary.basis for boundary in row.boundaries])
            values.append([boundary.value for boundary in row.boundaries])
            counts.append(len(row.boundaries))
            root_rows.append(_root_row(root, synthesized, presentation))
        champion_rows.append(_champion_row(root, synthesized))
    if not boards:
        raise ValueError("A sequence shard cannot be empty")
    directory.mkdir(parents=True)
    _array(
        directory,
        "board_tokens",
        boards,
        max(map(len, boards)),
        np.int16,
        PAD,
    )
    _array(
        directory,
        "trace_tokens",
        traces,
        max(map(len, traces)),
        np.int16,
        PAD,
    )
    _one_dimensional(directory, "trace_len", lengths, np.int32)
    _array(
        directory,
        "trace_roles",
        roles,
        max(map(len, roles)),
        np.int8,
        0,
    )
    for name, rows in [
        ("state_no_tool", no_tool),
        ("state_tool", tool),
        ("state_best", best),
        ("state_basis", basis),
        ("state_value", values),
    ]:
        _array(directory, name, rows, max(map(len, rows)), np.int32, -1)
    _one_dimensional(directory, "state_count", counts, np.int32)
    _write_jsonl(directory / "roots.jsonl", root_rows)
    _write_jsonl(directory / "champions.jsonl", champion_rows)
    files = {
        path.name: sha256(path)
        for path in sorted(
            list(directory.glob("*.npy"))
            + [directory / "roots.jsonl", directory / "champions.jsonl"]
        )
    }
    payload = {**metadata, "rows": len(boards), "files": files}
    write_manifest(directory / "manifest.json", payload)
    return payload


def _presentations_for(recipe, seed, root_id):
    presentation = sample_presentation(recipe, seed, root_id)
    if presentation is None:
        return PRESENTATIONS
    return (presentation,)


def _expected_root_rows(roots, presentations):
    return {
        (root["root_id"], presentation)
        for root in roots
        for presentation in presentations[root["root_id"]]
    }


def _recover_shard(output, index, roots, metadata, presentations):
    final = output / f"shard-{index:06d}"
    manifest = load_manifest(final / "manifest.json")
    if not isinstance(manifest.get("rows"), int):
        raise FileExistsError(f"Unregistered shard has no valid row count: {final}")
    published = {
        "path": final.name,
        "rows": manifest["rows"],
        "sha256": sha256(final / "manifest.json"),
    }
    _, rows, _ = verify_shard_files(output, published, metadata)
    if {(row["root_id"], row["presentation"]) for row in rows} != _expected_root_rows(
        roots, presentations
    ):
        raise FileExistsError(f"Unregistered shard has different roots: {final}")
    expected = {root["root_id"]: root for root in roots}
    for row in rows:
        root = expected[row["root_id"]]
        for name in (*ROOT_ROW_FIELDS, "teacher_width", "depth"):
            if name in root and row.get(name) != root[name]:
                raise FileExistsError(
                    f"Unregistered shard root {name} differs: {final}"
                )
    return published


def _publish_shard(output, index, records, metadata, presentations):
    final = output / f"shard-{index:06d}"
    temporary = output / f".shard-{index:06d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    payload = _write_shard(temporary, records, metadata, presentations)
    if final.exists():
        shutil.rmtree(temporary)
        roots = [record.root for record in records]
        return _recover_shard(output, index, roots, metadata, presentations)
    os.replace(temporary, final)
    return {
        "path": final.name,
        "rows": payload["rows"],
        "sha256": sha256(final / "manifest.json"),
    }


def _release_metadata(identity):
    return {
        "format_version": SEQUENCE_FORMAT_VERSION,
        "grammar_version": TRACE_GRAMMAR_VERSION,
        "kind": "teacher-sequence",
        "role_count": ROLE_COUNT,
        "tokenizer_version": CHESS_TOKENIZER_VERSION,
        **identity,
    }


def _sidecar_identity(sidecar_dir, roots):
    mapping_path = sidecar_dir / "sidecars.json"
    mapping = load_manifest(mapping_path)
    if not mapping:
        mapping = {}
        for root in roots:
            path = sidecar_dir / "roots" / root["root_id"] / "sidecar.jsonl"
            mapping[root["root_id"]] = sha256(path)
    if set(mapping) != {root["root_id"] for root in roots}:
        raise ValueError("Sidecar manifest does not cover exactly the selected roots")
    for root in roots:
        path = sidecar_dir / "roots" / root["root_id"] / "sidecar.jsonl"
        if sha256(path) != mapping[root["root_id"]]:
            raise ValueError(f"Sidecar checksum mismatch: {path}")
    return {"rows": len(mapping), "sha256": content_sha256(mapping)}


def _verify_identity(identity, roots, sidecar_dir, config, contract_path):
    required = {
        "teacher",
        "engine",
        "config",
        "search",
        "board",
        "presentation",
        "synthesis",
        "root_selection",
    }
    missing = required - set(identity)
    if missing:
        raise ValueError(f"Release identity is missing {sorted(missing)[0]}")
    if identity["search"] != config.search_dict():
        raise ValueError("Release search settings differ from the configuration")
    if identity["board"] != config.board_dict():
        raise ValueError("Release BOARD settings differ from the configuration")
    if identity["presentation"] != config.presentation_dict():
        raise ValueError("Release presentation settings differ from the configuration")
    if identity["synthesis"] != config.synthesis_dict():
        raise ValueError("Release synthesis settings differ from the configuration")
    if identity["config"].get("sha256") != sha256(config.source_path):
        raise ValueError("Release configuration checksum differs from the file")
    if config.recipe is None:
        if "recipe" in identity:
            raise ValueError("Release identity has a recipe the configuration omits")
    else:
        if identity.get("recipe") != config.recipe.identity():
            raise ValueError("Release recipe differs from the configuration")
        missing = [
            root["root_id"]
            for root in roots
            if "teacher_width" not in root or "depth" not in root
        ]
        if missing:
            raise ValueError(f"Release root has no recipe sample: {missing[0]}")
    widths = [root.get("teacher_width", config.search.teacher_width) for root in roots]
    expected_schema = SIDECAR_SCHEMA_WIDE if max(widths) > 1 else SIDECAR_SCHEMA
    if identity["teacher"].get("schema_version") != expected_schema:
        raise ValueError("Release teacher schema is unsupported")
    if identity["teacher"].get("contract_sha256") != sha256(contract_path):
        raise ValueError("Release teacher contract checksum differs from the file")
    actual_roots = {"rows": len(roots), "sha256": content_sha256(roots)}
    if "roots" in identity and identity["roots"] != actual_roots:
        raise ValueError("Release root identity differs from the selected roots")
    actual_sidecars = _sidecar_identity(sidecar_dir, roots)
    if "sidecars" in identity and identity["sidecars"] != actual_sidecars:
        raise ValueError("Release sidecar identity differs from the run")
    return {**identity, "roots": actual_roots, "sidecars": actual_sidecars}


def _synthesize_one(payload):
    root, path, config, seed = payload
    sidecar = load_sidecar(path)
    return synthesize_root(sidecar, root, config, seed=seed)


def _init_worker(config, seed):
    global _WORKER_CONFIG, _WORKER_SEED
    _WORKER_CONFIG = config
    _WORKER_SEED = seed


def _worker(payload):
    root, path = payload
    return _synthesize_one((root, path, _WORKER_CONFIG, _WORKER_SEED))


def _validate_completed_release(output, identity, roots, metadata, presentations):
    release_path = output / "release.json"
    if not release_path.exists():
        return set(), []
    previous = load_manifest(release_path)
    if previous.get("metadata") != identity or any(
        previous.get(key) != value for key, value in metadata.items()
    ):
        raise ValueError("The existing release belongs to a different configuration")
    complete = set(previous.get("complete_root_ids", []))
    shards = previous["shards"]
    expected = {
        (root_id, presentation)
        for root_id in complete
        for presentation in presentations[root_id]
    }
    published = Counter()
    for shard in shards:
        _, rows, _ = verify_shard_files(output, shard, metadata)
        published.update((row["root_id"], row["presentation"]) for row in rows)
    if set(published) != expected or any(count != 1 for count in published.values()):
        raise ValueError("Published shards do not match the release root list")
    known = {root["root_id"] for root in roots}
    if not complete <= known:
        raise ValueError("Release contains roots outside the selection")
    return complete, shards


def produce_shards(
    roots,
    sidecar_dir,
    output_dir,
    config,
    identity,
    contract_path=DEFAULT_TEACHER_PATH,
    shard_size=1024,
    workers=None,
):
    """Synthesize roots into independently published, resumable shards."""
    if shard_size < 1:
        raise ValueError("Shard size must be positive")
    roots = list(roots)
    if not roots:
        raise ValueError("A sequence release needs at least one root")
    if len({root["root_id"] for root in roots}) != len(roots):
        raise ValueError("A sequence release cannot contain duplicate root ids")
    if any(root.get("split") not in {"train", "val"} for root in roots):
        raise ValueError("Every sequence root must belong to train or val")
    identity = _verify_identity(
        identity, roots, Path(sidecar_dir), config, Path(contract_path)
    )
    collection_seed = identity["root_selection"]["seed"]
    presentations = {
        root["root_id"]: _presentations_for(
            config.recipe, collection_seed, root["root_id"]
        )
        for root in roots
    }
    release_metadata = _release_metadata(identity)
    shard_metadata = {
        key: value
        for key, value in release_metadata.items()
        if key not in RELEASE_ONLY_METADATA
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    release_path = output_dir / "release.json"
    complete, shards = _validate_completed_release(
        output_dir, identity, roots, release_metadata, presentations
    )
    pending = [root for root in roots if root["root_id"] not in complete]
    sidecar_dir = Path(sidecar_dir)
    seed = config.smoke.seed
    workers = config.search.workers if workers is None else workers
    if workers < 1:
        raise ValueError("Workers must be positive")
    profile = RootProfile(presentations)
    for start in range(0, len(pending), shard_size):
        group = pending[start : start + shard_size]
        index = len(shards)
        final = output_dir / f"shard-{index:06d}"
        if final.exists():
            shard = _recover_shard(
                output_dir, index, group, shard_metadata, presentations
            )
            records = []
        else:
            payloads = [
                (
                    root,
                    str(sidecar_dir / "roots" / root["root_id"] / "sidecar.jsonl"),
                )
                for root in group
            ]
            if workers == 1:
                records = [
                    _synthesize_one((root, path, config, seed))
                    for root, path in payloads
                ]
            else:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_worker,
                    initargs=(config, seed),
                ) as pool:
                    records = list(pool.map(_worker, payloads))
            shard = _publish_shard(
                output_dir, index, records, shard_metadata, presentations
            )
            for record in records:
                profile.add(record)
        shards.append(shard)
        complete.update(root["root_id"] for root in group)
        write_manifest(
            release_path,
            {
                **release_metadata,
                "metadata": identity,
                "shards": shards,
                "complete_root_ids": sorted(complete),
            },
        )
    statistics = profile.result() if profile.roots else None
    if statistics is not None:
        statistics["complete"] = profile.roots == len(roots)
        write_manifest(output_dir / "statistics.json", statistics)
    return {
        "shards": shards,
        "roots": len(complete),
        "release": str(release_path),
        "statistics": statistics,
    }


def load_collection(path):
    path = Path(path)
    collection = load_manifest(path)
    if not collection:
        raise FileNotFoundError(path)
    base = path.parent
    roots_path = base / collection["roots"]["path"]
    roots = [
        json.loads(line)
        for line in roots_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(roots) != collection["roots"]["rows"]:
        raise ValueError("Collection roots do not match their manifest")
    if content_sha256(roots) != collection["roots"]["sha256"]:
        raise ValueError("Collection roots do not match their declared hash")
    identity = {
        key: value
        for key, value in collection.items()
        if key
        in {
            "teacher",
            "engine",
            "config",
            "search",
            "board",
            "presentation",
            "synthesis",
            "root_selection",
            "roots",
            "sidecars",
            "recipe",
        }
    }
    identity["roots"] = {
        "rows": collection["roots"]["rows"],
        "sha256": collection["roots"]["sha256"],
    }
    return collection, roots, identity, base


def main():
    import argparse

    from pondera.data.trace_config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/trace_v2.toml", type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER_PATH, type=Path)
    parser.add_argument("--shard-size", type=int, default=1024)
    args = parser.parse_args()
    config = load_config(args.config)
    collection, roots, identity, base = load_collection(args.collection)
    output = args.output or (base / "release")
    result = produce_shards(
        roots,
        base,
        output,
        config,
        identity,
        contract_path=args.teacher,
        shard_size=args.shard_size,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "statistics"},
            indent=2,
        )
    )
    if result["statistics"] is not None:
        print(json.dumps(result["statistics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
