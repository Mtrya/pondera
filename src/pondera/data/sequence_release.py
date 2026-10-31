"""Compact, sharded Sequence teacher releases for large offline synthesis."""

import hashlib
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import numpy as np

from pondera.chess.codec import encode_board, mate_token, value_bin
from pondera.chess.symmetry import reflect_uci
from pondera.chess.vocabulary import (
    CHESS_TOKENIZER_VERSION,
    MOVE,
    PAD,
    UCI_MOVE_TO_IDX,
    VALUE,
)
from pondera.data.annotation import AnnotationConfig
from pondera.data.sequence_validation import (
    SEQUENCE_FORMAT_VERSION,
    shard_configuration,
    verify_shard_files,
)
from pondera.data.storage import load_manifest, write_manifest
from pondera.data.teacher_sequence import (
    DEFAULT_TEACHER_SPECS,
    TEACHER_BY_NAME,
    synthesize_root,
)
from pondera.data.trace import ROLE_COUNT, TRACE_GRAMMAR_VERSION, replay

BASIS_IDS = {"shallow": 0, "proposed": 1, "completed": 2}


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


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


def _state_value(state):
    if state["value"] is None:
        return -1
    if state["mate"] is not None:
        return mate_token(state["mate"], state["mate_win"])
    return VALUE + value_bin(state["value"])


def _teacher_rows(root, teacher, board):
    operations = teacher["operations"]
    final = operations[-1]
    root_black = chess.Board(root["fen"]).turn == chess.BLACK
    final_move = final["uci"] if final["op"] == "final" else None
    if final_move is None:
        raise ValueError("Teacher trace has no final move")
    states = teacher["truncation"]["states"]
    return {
        "board": board,
        "trace": teacher["tool"]["tokens"],
        "roles": replay(
            teacher["tool"]["tokens"], root["fen"], root.get("repetition", 1)
        ).roles,
        "trace_len": len(teacher["tool"]["tokens"]),
        "state_no_tool": [state["no_tool"] for state in states],
        "state_tool": [state["tool"] for state in states],
        "state_best": [
            MOVE
            + UCI_MOVE_TO_IDX[
                reflect_uci(state["best"]) if root_black else state["best"]
            ]
            for state in states
        ],
        "state_basis": [BASIS_IDS[state["basis"]] for state in states],
        "state_value": [_state_value(state) for state in states],
        "state_count": len(states),
    }


def _write_shard(directory, records, metadata):
    boards, traces, roles, lengths = [], [], [], []
    no_tool, tool, best, basis, values, counts = [], [], [], [], [], []
    root_rows = []
    for record in records:
        board = encode_board(record["fen"], record.get("repetition", 1))
        for name in metadata["teachers"]:
            teacher = record["teachers"][name]
            row = _teacher_rows(record, teacher, board)
            boards.append(row.pop("board"))
            traces.append(row.pop("trace"))
            roles.append(row.pop("roles"))
            lengths.append(row.pop("trace_len"))
            no_tool.append(row.pop("state_no_tool"))
            tool.append(row.pop("state_tool"))
            best.append(row.pop("state_best"))
            basis.append(row.pop("state_basis"))
            values.append(row.pop("state_value"))
            counts.append(row.pop("state_count"))
            root_rows.append(
                {
                    "root_id": record["root_id"],
                    "split": record["split"],
                    "fen": record["fen"],
                    "repetition": record.get("repetition", 1),
                    "turn": record["turn"],
                    "teacher": name,
                    "d16": record.get("d16"),
                    "source": record.get("source"),
                }
            )
    if not boards:
        raise ValueError("A sequence shard cannot be empty")
    directory.mkdir(parents=True)
    _array(directory, "board_tokens", boards, max(map(len, boards)), np.int16, PAD)
    _array(directory, "trace_tokens", traces, max(map(len, traces)), np.int16, PAD)
    _one_dimensional(directory, "trace_len", lengths, np.int16)
    _array(directory, "trace_roles", roles, max(map(len, roles)), np.int8, 0)
    for name, rows in [
        ("state_no_tool", no_tool),
        ("state_tool", tool),
        ("state_best", best),
        ("state_basis", basis),
        ("state_value", values),
    ]:
        _array(directory, name, rows, max(map(len, rows)), np.int16, -1)
    _one_dimensional(directory, "state_count", counts, np.int16)
    _write_jsonl(directory / "roots.jsonl", root_rows)
    files = {
        path.name: _sha256(path)
        for path in sorted(list(directory.glob("*.npy")) + [directory / "roots.jsonl"])
    }
    payload = {
        **metadata,
        "rows": len(boards),
        "files": files,
    }
    write_manifest(directory / "manifest.json", payload)
    return payload


def _publish_shard(output, index, records, metadata):
    final = output / f"shard-{index:06d}"
    temporary = output / f".shard-{index:06d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    payload = _write_shard(temporary, records, metadata)
    if final.exists():
        existing = load_manifest(final / "manifest.json")
        if shard_configuration(existing) != shard_configuration(payload):
            raise FileExistsError(f"Unregistered shard has different contents: {final}")
        published = {
            "path": final.name,
            "rows": payload["rows"],
            "sha256": _sha256(final / "manifest.json"),
        }
        verify_shard_files(output, published, metadata)
        shutil.rmtree(temporary)
    else:
        os.replace(temporary, final)
    return {
        "path": final.name,
        "rows": payload["rows"],
        "sha256": _sha256(final / "manifest.json"),
    }


def _release_metadata(metadata):
    return {
        "format_version": SEQUENCE_FORMAT_VERSION,
        "grammar_version": TRACE_GRAMMAR_VERSION,
        "tokenizer_version": CHESS_TOKENIZER_VERSION,
        "kind": "teacher-sequence",
        "role_count": ROLE_COUNT,
        **metadata,
    }


def _release_identity(roots, config, metadata):
    roots = list(roots)
    root_ids = [root["root_id"] for root in roots]
    if len(set(root_ids)) != len(root_ids):
        raise ValueError("A sequence release cannot contain duplicate root ids")
    teachers = tuple(
        metadata.get("teachers", [spec.name for spec in DEFAULT_TEACHER_SPECS])
    )
    if not teachers or len(set(teachers)) != len(teachers):
        raise ValueError("Release teacher names must be nonempty and unique")
    root_json = json.dumps(roots, sort_keys=True, separators=(",", ":"))
    return {
        **metadata,
        "teachers": list(teachers),
        "teacher_specs": {
            spec.name: spec.as_dict() for spec in teacher_specs(teachers)
        },
        "search": {
            name: getattr(config, name)
            for name in ("depth", "multipv", "threads", "hash_mb", "max_time")
        },
        "roots": {
            "rows": len(roots),
            "sha256": hashlib.sha256(root_json.encode("utf-8")).hexdigest(),
        },
    }


def produce_shards(roots, output_dir, config, metadata, shard_size=1024):
    """Synthesize roots into independently published, resumable data shards."""
    if shard_size < 1:
        raise ValueError("Shard size must be positive")
    roots = list(roots)
    metadata = _release_identity(roots, config, metadata)
    engine_path = shutil.which(config.engine)
    if engine_path is None:
        raise FileNotFoundError(config.engine)
    metadata["engine"] = {
        **metadata.get("engine", {}),
        "path": engine_path,
        "sha256": _sha256(engine_path),
    }
    release_metadata = _release_metadata(metadata)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    release_path = output_dir / "release.json"
    if release_path.exists():
        previous = load_manifest(release_path)
        if previous.get("metadata") != metadata or any(
            previous.get(key) != value for key, value in release_metadata.items()
        ):
            raise ValueError(
                "The existing release belongs to a different configuration"
            )
        complete = set(previous.get("complete_root_ids", []))
        shards = previous["shards"]
        published = Counter()
        for shard in shards:
            _, rows, _ = verify_shard_files(output_dir, shard, release_metadata)
            published.update((row["root_id"], row["teacher"]) for row in rows)
        expected = Counter(
            (root_id, teacher)
            for root_id in complete
            for teacher in metadata["teachers"]
        )
        if published != expected:
            raise ValueError("Published shards do not match the release root list")
    else:
        complete = set()
        shards = []
    pending = [root for root in roots if root["root_id"] not in complete]
    with ProcessPoolExecutor(
        max_workers=config.workers,
        initializer=_initialize_worker,
        initargs=(config, metadata["teachers"]),
    ) as pool:
        for start in range(0, len(pending), shard_size):
            group = pending[start : start + shard_size]
            records = list(pool.map(_synthesize_root, group))
            index = len(shards)
            shard = _publish_shard(output_dir, index, records, release_metadata)
            shards.append(shard)
            complete.update(root["root_id"] for root in group)
            payload = {
                **release_metadata,
                "metadata": metadata,
                "shards": shards,
                "complete_root_ids": sorted(complete),
            }
            write_manifest(release_path, payload)
    return {
        "shards": shards,
        "roots": len(complete),
        "release": str(release_path),
    }


class _SynthesisWorker:
    def __init__(self, config, teachers):
        self.config = config
        self.annotation = config.annotation()
        self.teachers = teachers

    def synthesize(self, root):
        engine_path = shutil.which(self.config.engine)
        if engine_path is None:
            raise FileNotFoundError(self.config.engine)
        engine = chess.engine.SimpleEngine.popen_uci(engine_path, timeout=60)
        try:
            engine.configure(
                {"Threads": self.config.threads, "Hash": self.config.hash_mb}
            )
            return synthesize_root(engine, root, self.annotation, self.teachers)
        finally:
            try:
                engine.quit()
            except chess.engine.EngineError:
                pass


_synthesis_worker = None


def _initialize_worker(config, teachers):
    global _synthesis_worker
    _synthesis_worker = _SynthesisWorker(config, teacher_specs(teachers))


def _synthesize_root(root):
    return _synthesis_worker.synthesize(root)


def merge_releases(inputs, output_dir):
    """Publish independently produced shards as one read-only release view."""
    source_shards = []
    complete = set()
    metadata = None
    selections = []
    root_sets = []
    for directory in inputs:
        directory = Path(directory).resolve()
        manifest = load_manifest(directory / "release.json")
        part_metadata = manifest["metadata"]
        release_metadata = _release_metadata(part_metadata)
        if any(manifest.get(key) != value for key, value in release_metadata.items()):
            raise ValueError(f"Release metadata differs from its identity: {directory}")
        configuration = {
            key: value
            for key, value in part_metadata.items()
            if key not in {"root_selection", "root_selections", "roots", "root_sets"}
        }
        if metadata is None:
            metadata = configuration
        elif configuration != metadata:
            raise ValueError("Cannot merge different teacher configurations")
        if "root_selection" in part_metadata:
            selections.append(part_metadata["root_selection"])
        selections.extend(part_metadata.get("root_selections", []))
        if "roots" in part_metadata:
            root_sets.append(part_metadata["roots"])
        root_sets.extend(part_metadata.get("root_sets", []))
        root_ids_list = manifest.get("complete_root_ids", [])
        root_ids = set(root_ids_list)
        if len(root_ids) != len(root_ids_list):
            raise ValueError(f"Release contains duplicate root ids: {directory}")
        duplicates = complete.intersection(root_ids)
        if duplicates:
            raise ValueError(f"Cannot merge duplicate roots: {sorted(duplicates)}")
        published = Counter()
        for shard in manifest["shards"]:
            _, rows, _ = verify_shard_files(directory, shard, release_metadata)
            published.update((row["root_id"], row["teacher"]) for row in rows)
            source_shards.append((directory, shard))
        expected = Counter(
            (root_id, teacher)
            for root_id in root_ids
            for teacher in part_metadata["teachers"]
        )
        if published != expected:
            raise ValueError(f"Release shards do not match its root list: {directory}")
        complete.update(root_ids)
    if metadata is None:
        raise ValueError("No releases to merge")
    if selections:
        metadata["root_selections"] = selections
    if root_sets:
        metadata["root_sets"] = root_sets

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Merge output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for directory, shard in source_shards:
        source = directory / shard["path"]
        target = output_dir / f"shard-{len(shards):06d}"
        target.symlink_to(source)
        shards.append(
            {
                "path": target.name,
                "rows": shard["rows"],
                "sha256": _sha256(target / "manifest.json"),
            }
        )
    write_manifest(
        output_dir / "release.json",
        {
            **_release_metadata(metadata),
            "metadata": metadata,
            "shards": shards,
            "complete_root_ids": sorted(complete),
        },
    )
    return {"shards": shards, "roots": len(complete)}


@dataclass(frozen=True)
class SequenceReleaseConfig:
    engine: str = "stockfish"
    depth: int = 8
    multipv: int = 4
    threads: int = 1
    hash_mb: int = 64
    max_time: float | None = None
    workers: int = 8

    def annotation(self):
        return AnnotationConfig(
            engine=self.engine,
            depth=self.depth,
            multipv=self.multipv,
            threads=self.threads,
            hash_mb=self.hash_mb,
            max_time=self.max_time,
        )


def teacher_specs(names):
    return tuple(TEACHER_BY_NAME[name] for name in names)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--engine", default="stockfish")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument(
        "--teachers",
        nargs="+",
        choices=sorted(TEACHER_BY_NAME),
        default=[spec.name for spec in DEFAULT_TEACHER_SPECS],
    )
    args = parser.parse_args()
    if args.start < 0 or (args.end is not None and args.end <= args.start):
        raise ValueError("Invalid root range")
    with args.roots.open(encoding="utf-8") as handle:
        roots = [json.loads(line) for line in handle if line.strip()]
    selected = roots[args.start : args.end]
    config = SequenceReleaseConfig(
        engine=args.engine,
        depth=args.depth,
        multipv=args.multipv,
        threads=args.threads,
        hash_mb=args.hash_mb,
        workers=args.workers,
    )
    metadata = {
        "teachers": args.teachers,
        "engine": {
            "depth": config.depth,
            "multipv": config.multipv,
            "threads": config.threads,
            "hash_mb": config.hash_mb,
            "max_time": config.max_time,
        },
        "root_selection": {
            "path": str(args.roots),
            "start": args.start,
            "end": args.end,
        },
    }
    result = produce_shards(selected, args.output, config, metadata, args.shard_size)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
