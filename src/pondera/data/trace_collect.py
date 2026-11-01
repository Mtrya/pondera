"""Collect instrumented Stockfish search records for grammar-v2 traces.

The collector samples roots from an existing root release, starts one fresh
engine process per root, and writes the sidecar and its root metadata. The
collection manifest is the identity the release builder consumes; resuming
reuses valid sidecars and re-runs missing or invalid ones.
"""

import json
import random
import select
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess

from pondera.data.storage import content_sha256, load_manifest, sha256, write_manifest
from pondera.data.trace_config import sample_depth, sample_width
from pondera.data.trace_synthesis import (
    SIDECAR_SCHEMA,
    SIDECAR_SCHEMA_WIDE,
    load_sidecar,
    validate_root,
)

SELECTION_ALGORITHM = (
    "random.Random(f'{seed}:{split}') reservoir sampling in source order; "
    "train roots whose position key occurs in validation are excluded"
)
SPLITS = ("train", "val")


class CollectionError(RuntimeError):
    """A collection that cannot be reproduced from its inputs."""


def _read_lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise CollectionError(
                    f"Invalid root JSON on line {index + 1}"
                ) from error
            if not isinstance(record, dict):
                raise CollectionError(
                    f"Root record on line {index + 1} is not an object"
                )
            yield index, record


def _position_key(fen):
    return " ".join(fen.split()[:4])


def _basic_record(record, index):
    root_id = record.get("root_id")
    split = record.get("split")
    fen = record.get("fen")
    repetition = record.get("repetition")
    if not isinstance(root_id, str) or not root_id:
        raise CollectionError(f"Root on line {index + 1} has no root_id")
    if split not in SPLITS:
        raise CollectionError(f"Root {root_id} has an unknown split")
    if not isinstance(fen, str) or len(fen.split()) != 6:
        raise CollectionError(f"Root {root_id} has an invalid FEN")
    if not isinstance(repetition, int) or isinstance(repetition, bool):
        raise CollectionError(f"Root {root_id} has an invalid repetition count")
    position_key = record.get("position_key", _position_key(fen))
    if position_key != _position_key(fen):
        raise CollectionError(f"Root {root_id} has an inconsistent position key")
    return root_id, split, fen, repetition, position_key


def _validated_root(record, index):
    root_id, split, fen, repetition, position_key = _basic_record(record, index)
    if not 1 <= repetition <= 4:
        raise CollectionError(f"Root {root_id} has an excluded repetition count")
    board = chess.Board(fen)
    if board.is_checkmate() or board.is_stalemate() or not board.legal_moves:
        raise CollectionError(f"Root {root_id} has no legal move")
    turn = "white" if board.turn == chess.WHITE else "black"
    if record.get("turn") not in (None, turn):
        raise CollectionError(f"Root {root_id} has a turn inconsistent with its FEN")
    result = {
        "root_id": root_id,
        "split": split,
        "fen": fen,
        "repetition": repetition,
        "turn": turn,
        "position_key": position_key,
    }
    for name in ("source", "d16", "provenance"):
        if name in record:
            result[name] = record[name]
    return result


def _allocate(total, counts):
    """Allocate a total over splits in proportion to eligible counts."""
    available = sum(counts.values())
    total = min(total, available)
    if total == 0:
        return {split: 0 for split in counts}
    quotas = {split: total * counts[split] / available for split in counts}
    allocation = {split: int(quotas[split]) for split in counts}
    remainder = total - sum(allocation.values())
    order = sorted(
        counts,
        key=lambda split: (quotas[split] - allocation[split], counts[split]),
        reverse=True,
    )
    for split in order[:remainder]:
        allocation[split] += 1
    for split in counts:
        if (
            counts[split]
            and allocation[split] == 0
            and total >= len([name for name in counts if counts[name]])
        ):
            donor = max(allocation, key=allocation.get)
            if allocation[donor] > 1:
                allocation[donor] -= 1
                allocation[split] += 1
    return allocation


def select_roots(source, requested, seed):
    """Sample train/val roots reproducibly and return their metadata."""
    source = Path(source)
    if requested < 1:
        raise CollectionError("The requested root count must be positive")
    validation_keys = set()
    val_count = 0
    seen = set()
    for index, record in _read_lines(source):
        root_id, split, fen, repetition, position_key = _basic_record(record, index)
        if root_id in seen:
            raise CollectionError(f"Duplicate root id in the source: {root_id}")
        seen.add(root_id)
        if not 1 <= repetition <= 4:
            continue
        if split == "val":
            validation_keys.add(position_key)
            val_count += 1
    train_count = 0
    for index, record in _read_lines(source):
        root_id, split, fen, repetition, position_key = _basic_record(record, index)
        if not 1 <= repetition <= 4:
            continue
        if split == "train" and position_key not in validation_keys:
            train_count += 1
    counts = {"train": train_count, "val": val_count}
    targets = _allocate(requested, counts)
    reservoirs = {split: [] for split in SPLITS}
    randoms = {split: random.Random(f"{seed}:{split}") for split in SPLITS}
    eligible_seen = {split: 0 for split in SPLITS}
    for index, record in _read_lines(source):
        root_id, split, fen, repetition, position_key = _basic_record(record, index)
        if not 1 <= repetition <= 4:
            continue
        if split == "train" and position_key in validation_keys:
            continue
        target = targets[split]
        if target == 0:
            continue
        eligible_seen[split] += 1
        reservoir = reservoirs[split]
        if len(reservoir) < target:
            reservoir.append((index, record))
        else:
            position = randoms[split].randrange(eligible_seen[split])
            if position < target:
                reservoir[position] = (index, record)
    selected = []
    for split in SPLITS:
        selected.extend(
            _validated_root(record, index)
            for index, record in sorted(reservoirs[split], key=lambda item: item[0])
        )
    if len(selected) != min(requested, sum(counts.values())):
        raise CollectionError("Root selection did not produce the requested sample")
    ids = {root["root_id"] for root in selected}
    if len(ids) != len(selected):
        raise CollectionError("Selected roots contain duplicate ids")
    train_keys = {root["position_key"] for root in selected if root["split"] == "train"}
    val_keys = {root["position_key"] for root in selected if root["split"] == "val"}
    if train_keys & val_keys:
        raise CollectionError("Train and validation roots overlap by position")
    selection = {
        "source": str(source),
        "source_sha256": sha256(source),
        "algorithm": SELECTION_ALGORITHM,
        "seed": seed,
        "requested": requested,
        "counts": {
            split: sum(1 for root in selected if root["split"] == split)
            for split in SPLITS
        },
        "source_counts": counts,
    }
    return selected, selection


def _root_teacher(root, config):
    width = root.get("teacher_width", config.search.teacher_width)
    depth = root.get("depth", config.search.depth)
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise CollectionError(f"Root {root['root_id']} has an invalid teacher width")
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        raise CollectionError(f"Root {root['root_id']} has an invalid depth")
    return width, depth


def apply_recipe(roots, config, seed):
    """Attach each root's sampled teacher width and depth from the recipe."""
    if config.recipe is None:
        return list(roots)
    sampled = []
    for root in roots:
        sampled.append(
            {
                **root,
                "teacher_width": sample_width(
                    config.recipe, seed, root["root_id"], config.search.teacher_width
                ),
                "depth": sample_depth(
                    config.recipe, seed, root["root_id"], config.search.depth
                ),
            }
        )
    return sampled


def _teacher_schema(roots, config):
    widths = [root.get("teacher_width", config.search.teacher_width) for root in roots]
    return SIDECAR_SCHEMA_WIDE if max(widths) > 1 else SIDECAR_SCHEMA


def _engine_path(config):
    path = Path(config.engine.path)
    if path.is_file():
        return path.resolve()
    resolved = shutil.which(config.engine.path)
    if resolved is None:
        raise FileNotFoundError(config.engine.path)
    return Path(resolved).resolve()


def _check_sidecar(root, sidecar, config, engine_name, teacher=None):
    width, depth = _root_teacher(root, config)
    repetition = validate_root(root, sidecar)
    if sidecar.engine != engine_name:
        raise CollectionError(f"Engine name differs for root {root['root_id']}")
    if teacher is not None and (
        sidecar.value_mate != teacher[0] or sidecar.max_ply != teacher[1]
    ):
        raise CollectionError(f"Teacher constants differ for root {root['root_id']}")
    if sidecar.search_start["depth"] != depth:
        raise CollectionError(f"Requested depth differs for root {root['root_id']}")
    if sidecar.search_end["depth"] != depth:
        raise CollectionError(f"Search did not complete for root {root['root_id']}")
    if sidecar.session["options"]["threads"] != config.search.threads:
        raise CollectionError(f"Thread count differs for root {root['root_id']}")
    if sidecar.session["options"]["hash_mb"] != config.search.hash_mb:
        raise CollectionError(f"Hash size differs for root {root['root_id']}")
    if sidecar.configured_width != width:
        raise CollectionError(f"Teacher width differs for root {root['root_id']}")
    return repetition


def _run_engine(engine_path, fen, config, trace_path, width, depth, timeout=900):
    """Drive one engine over raw UCI in the contract's exact command order.

    python-chess manages MultiPV itself and refuses to configure it, so the
    collector speaks plain UCI: options first, TraceLog last, a readiness
    barrier, then position and a depth-limited go.
    """
    process = subprocess.Popen(
        [str(engine_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        bufsize=0,
    )

    def send(line):
        process.stdin.write((line + "\n").encode())
        process.stdin.flush()

    buffer = b""

    def read_line(deadline):
        nonlocal buffer
        while b"\n" not in buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CollectionError("engine timed out")
            ready, _, _ = select.select([process.stdout], [], [], remaining)
            if not ready:
                continue
            chunk = process.stdout.read(65536)
            if not chunk:
                raise CollectionError("engine closed its output")
            buffer += chunk
        line, buffer = buffer.split(b"\n", 1)
        return line.decode(errors="replace").strip()

    def read_until(prefix, deadline):
        seen = []
        while True:
            line = read_line(deadline)
            seen.append(line)
            if line.startswith(prefix):
                return line, seen

    deadline = time.monotonic() + timeout
    try:
        send("uci")
        _, banner = read_until("uciok", deadline)
        if not any(line.startswith("option name TraceLog ") for line in banner):
            raise CollectionError(f"Engine does not expose TraceLog: {engine_path}")
        send(f"setoption name Threads value {config.search.threads}")
        send(f"setoption name Hash value {config.search.hash_mb}")
        send(f"setoption name MultiPV value {width}")
        send(f"setoption name TraceLog value {trace_path}")
        send("isready")
        read_until("readyok", deadline)
        send(f"position fen {fen}")
        send(f"go depth {depth}")
        bestmove, _ = read_until("bestmove", deadline)
    finally:
        try:
            send("quit")
        except (BrokenPipeError, ValueError):
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    parts = bestmove.split()
    if len(parts) < 2:
        raise CollectionError(f"engine returned no bestmove: {bestmove!r}")
    return parts[1]


def collect_root(root, output, config, engine_path, engine_name=None, teacher=None):
    """Run one fresh engine process and return its validated root metadata."""
    directory = Path(output) / "roots" / root["root_id"]
    directory.mkdir(parents=True, exist_ok=True)
    sidecar_path = directory / "sidecar.jsonl"
    temporary = directory / ".sidecar.jsonl.tmp"
    if temporary.exists():
        temporary.unlink()
    width, depth = _root_teacher(root, config)
    bestmove = _run_engine(engine_path, root["fen"], config, temporary, width, depth)
    if not temporary.exists():
        raise CollectionError(f"Engine wrote no sidecar for root {root['root_id']}")
    sidecar = load_sidecar(temporary, teacher_width=width)
    if engine_name is not None and sidecar.engine != engine_name:
        raise CollectionError(f"Engine name differs for root {root['root_id']}")
    _check_sidecar(root, sidecar, config, sidecar.engine, teacher)
    if sidecar.search_end["bestmove"] != bestmove:
        raise CollectionError(
            f"bestmove differs from the sidecar for {root['root_id']}"
        )
    temporary.replace(sidecar_path)
    meta = {
        **root,
        "sidecar": sidecar_path.name,
        "sidecar_sha256": sha256(sidecar_path),
        "bestmove": bestmove,
        "engine_sha256": sha256(engine_path),
        "config_sha256": sha256(config.source_path),
        "contract_sha256": None,
        "value_mate": sidecar.value_mate,
        "max_ply": sidecar.max_ply,
        "schema_version": SIDECAR_SCHEMA if width == 1 else SIDECAR_SCHEMA_WIDE,
    }
    write_manifest(directory / "meta.json", meta)
    return meta


def _collection_path(output):
    return Path(output) / "collection.json"


def _roots_path(output):
    return Path(output) / "roots.jsonl"


def _sidecars_path(output):
    return Path(output) / "sidecars.json"


def _write_roots(output, roots):
    path = _roots_path(output)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for root in roots:
            handle.write(json.dumps(root, separators=(",", ":")) + "\n")
    temporary.replace(path)
    return {"path": path.name, "rows": len(roots), "sha256": content_sha256(roots)}


def _selection_record(root):
    return {
        key: value
        for key, value in root.items()
        if key not in {"teacher_width", "depth"}
    }


def _load_roots(output, selection, roots):
    path = _roots_path(output)
    if path.exists():
        existing = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if content_sha256(
            [_selection_record(root) for root in existing]
        ) != content_sha256([_selection_record(root) for root in roots]):
            raise CollectionError("Existing root selection differs from this run")
        return {
            "path": path.name,
            "rows": len(existing),
            "sha256": content_sha256(existing),
        }, existing
    return _write_roots(output, roots), roots


def _static_identity(config, selection, roots_identity, engine, teacher):
    identity = {
        "teacher": teacher,
        "engine": engine,
        "config": {
            "path": str(config.source_path),
            "sha256": sha256(config.source_path),
        },
        "search": config.search_dict(),
        "board": config.board_dict(),
        "presentation": config.presentation_dict(),
        "synthesis": config.synthesis_dict(),
        "root_selection": selection,
        "roots": roots_identity,
    }
    if config.recipe is not None:
        identity["recipe"] = config.recipe.identity()
    return identity


def _write_collection(output, identity, sidecars):
    payload = {
        "kind": "trace-v2-collection",
        "format_version": 1,
        **identity,
        "sidecars": sidecars,
    }
    write_manifest(_collection_path(output), payload)


def _load_collection(output):
    return load_manifest(_collection_path(output))


def collect(
    roots,
    output,
    config,
    instrumentation_path,
    control_path,
    contract_path,
    selection,
):
    """Collect sidecars for selected roots, resuming completed roots."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    engine_path = _engine_path(config)
    instrumentation_path = Path(instrumentation_path)
    control_path = Path(control_path)
    contract_path = Path(contract_path)
    for path in (engine_path, instrumentation_path, control_path, contract_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    sampled = apply_recipe(roots, config, selection["seed"])
    roots_identity, roots = _load_roots(output, selection, sampled)
    if config.recipe is not None:
        missing = [
            root["root_id"]
            for root in roots
            if "teacher_width" not in root or "depth" not in root
        ]
        if missing:
            raise CollectionError(
                f"Existing roots have no recipe samples: {missing[0]}"
            )
    roots_by_id = {root["root_id"]: root for root in roots}
    if len(roots_by_id) != len(roots):
        raise CollectionError("Selected roots contain duplicate ids")
    sidecars = load_manifest(_sidecars_path(output))
    for root_id, digest in list(sidecars.items()):
        path = output / "roots" / root_id / "sidecar.jsonl"
        if root_id not in roots_by_id or not path.exists() or sha256(path) != digest:
            del sidecars[root_id]
    previous = _load_collection(output)
    engine_identity = {
        "name": previous.get("engine", {}).get("name"),
        "sha256": sha256(engine_path),
        "instrumentation": str(instrumentation_path),
        "instrumentation_sha256": sha256(instrumentation_path),
        "control": {"path": str(control_path), "sha256": sha256(control_path)},
    }
    teacher = {
        "schema_version": _teacher_schema(roots, config),
        "contract": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "value_mate": previous.get("teacher", {}).get("value_mate"),
        "max_ply": previous.get("teacher", {}).get("max_ply"),
    }
    if previous:
        expected = _static_identity(
            config,
            selection,
            roots_identity,
            {
                **engine_identity,
                "name": previous["engine"]["name"],
            },
            {
                **teacher,
                "value_mate": previous["teacher"]["value_mate"],
                "max_ply": previous["teacher"]["max_ply"],
            },
        )
        for key, value in expected.items():
            if previous.get(key) != value:
                raise CollectionError(
                    f"Existing collection {key} differs from this run"
                )

    completed = set(sidecars)
    pending = [root for root in roots if root["root_id"] not in completed]
    first = None
    if previous:
        engine_identity["name"] = previous["engine"]["name"]
        teacher["value_mate"] = previous["teacher"]["value_mate"]
        teacher["max_ply"] = previous["teacher"]["max_ply"]
    elif pending:
        first = pending.pop(0)
        meta = collect_root(first, output, config, engine_path)
        sidecars[first["root_id"]] = meta["sidecar_sha256"]
        engine_identity["name"] = json.loads(
            (output / "roots" / first["root_id"] / "sidecar.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )["engine"]
        session = load_sidecar(output / "roots" / first["root_id"] / "sidecar.jsonl")
        teacher["value_mate"] = session.value_mate
        teacher["max_ply"] = session.max_ply
        if not teacher["value_mate"] or not teacher["max_ply"]:
            raise CollectionError("The first sidecar has no mate constants")
    elif sidecars:
        sample_id = next(iter(sidecars))
        sample = load_sidecar(output / "roots" / sample_id / "sidecar.jsonl")
        engine_identity["name"] = sample.engine
        teacher["value_mate"] = sample.value_mate
        teacher["max_ply"] = sample.max_ply
    if not engine_identity["name"]:
        raise CollectionError("The collection has no engine identity")
    if not teacher["value_mate"] or not teacher["max_ply"]:
        raise CollectionError("The collection has no teacher mate constants")
    identity = _static_identity(
        config, selection, roots_identity, engine_identity, teacher
    )
    _write_collection(
        output,
        identity,
        {
            "rows": len(sidecars),
            "sha256": content_sha256(sidecars),
        },
    )
    write_manifest(_sidecars_path(output), sidecars)

    def run(root):
        meta = collect_root(
            root,
            output,
            config,
            engine_path,
            engine_identity["name"],
            (teacher["value_mate"], teacher["max_ply"]),
        )
        return root["root_id"], meta["sidecar_sha256"]

    if pending:
        with ThreadPoolExecutor(max_workers=config.search.workers) as pool:
            for root_id, digest in pool.map(run, pending):
                sidecars[root_id] = digest
                write_manifest(_sidecars_path(output), sidecars)
                _write_collection(
                    output,
                    identity,
                    {"rows": len(sidecars), "sha256": content_sha256(sidecars)},
                )
    return {
        "roots": len(sidecars),
        "pending": len(pending),
        "collection": str(_collection_path(output)),
        "sidecars": str(_sidecars_path(output)),
    }


def main():
    import argparse

    from pondera.data.trace_config import DEFAULT_TEACHER_PATH, load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/trace_v2.toml", type=Path)
    parser.add_argument("--roots", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--instrumentation", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER_PATH, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    roots, selection = select_roots(args.roots, config.smoke.roots, config.smoke.seed)
    output = args.output or Path(config.smoke.output)
    result = collect(
        roots,
        output,
        config,
        args.instrumentation,
        args.control,
        args.teacher,
        selection,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
