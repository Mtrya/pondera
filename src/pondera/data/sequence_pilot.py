"""Root selection, resumable synthesis, reference quality, and pilot CLI.

The pilot freezes a subset of an existing annotation release, synthesizes the
three teacher trees over every selected root, measures the final answers on
the validation roots with a depth-16 restricted search, and writes checksummed
manifests and a summary. It never trains and never modifies the source release.

Every root is a single JSONL record produced by one worker with one Stockfish
process. Each worker reads all three teachers through one path-keyed node cache.
Records are appended only after their full tree and trace replay checks pass;
resume drops an unterminated or malformed final line and regenerates that root.

Command line (run directories live outside the package)::

    python -m pondera.data.sequence_pilot --plan PLAN.json --output DIR
    python -m pondera.data.sequence_pilot --plan PLAN.json --output DIR \\
        --train-roots 2 --val-roots 2 --workers 2

The plan is read, never written. All effective synthesis parameters are captured
in ``config.json`` and the per-stage identity files.
"""

import argparse
import json
import os
import random
import shutil
import threading
import time
import traceback
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import chess
import chess.engine

from pondera.chess.codec import encode_board
from pondera.chess.vocabulary import CHESS_TOKENIZER_VERSION
from pondera.data.annotation import AnnotationConfig
from pondera.data.storage import load_manifest, write_manifest
from pondera.data.teacher_sequence import (
    BOARD_PLY_INTERVAL,
    BOARD_RETURN_PLIES,
    MAX_ADAPTIVE_CANDIDATES,
    REQUIRED_DEPTHS,
    STABILITY_NEAR_VALUE,
    STABILITY_VALUE_FLOOR,
    TEACHER_BY_NAME,
    TEACHER_SPECS,
    VALUE_EPSILON,
    build_states,
    emit_operations,
    plan_boards,
    restricted_value,
    synthesize_root,
    verify_teacher_trace,
)
from pondera.data.trace import TRACE_GRAMMAR_VERSION

ROOT_FORMAT_VERSION = 1
SYNTHESIS_FORMAT_VERSION = 1
QUALITY_FORMAT_VERSION = 1
RUN_FORMAT_VERSION = 1

POSITION_KEY = "first four FEN fields"
ROOT_RNG = "random.Random(seed) per split, reservoir sampling in global row order"


def _canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _sha256_file(path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_sha(identity) -> str:
    return _sha256_bytes(_canonical(identity).encode("utf-8"))


def _rate(numerator, denominator):
    return None if not denominator else numerator / denominator


def _mean(total, count):
    return None if not count else total / count


@dataclass(frozen=True)
class PilotConfig:
    release: str
    release_sha256: str
    train_roots: int = 1024
    val_roots: int = 512
    seed: int = 640
    engine: str = "stockfish"
    engine_version: str | None = None
    depth: int = 8
    multipv: int = 4
    threads: int = 1
    hash_mb: int = 64
    max_time: float | None = None
    workers: int = 8
    teachers: tuple[str, ...] = tuple(spec.name for spec in TEACHER_SPECS)
    reference_depth: int = 16
    reference_val_rows: int = 512
    position_key: str = POSITION_KEY

    def __post_init__(self):
        object.__setattr__(self, "teachers", tuple(self.teachers))
        if not self.release:
            raise ValueError("A release manifest is required")
        if len(self.release_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.release_sha256
        ):
            raise ValueError("The release sha256 must be 64 lowercase hex characters")
        if min(self.train_roots, self.val_roots, self.workers) < 1:
            raise ValueError("Root counts and worker count must be positive")
        if self.seed < 0:
            raise ValueError("The seed must be nonnegative")
        if not self.engine:
            raise ValueError("An engine executable is required")
        if (
            min(
                self.depth,
                self.multipv,
                self.threads,
                self.hash_mb,
                self.reference_depth,
            )
            < 1
        ):
            raise ValueError("Engine limits must be positive")
        if self.reference_val_rows < 0:
            raise ValueError("The reference validation row count cannot be negative")
        if self.max_time is not None and self.max_time <= 0:
            raise ValueError("The time limit must be positive or None")
        if self.position_key != POSITION_KEY:
            raise ValueError(f"Unsupported position key: {self.position_key}")
        if len(set(self.teachers)) != len(self.teachers):
            raise ValueError("Teacher names must be unique")
        unknown = set(self.teachers) - set(TEACHER_BY_NAME)
        if unknown:
            raise ValueError(f"Unknown teachers: {sorted(unknown)}")

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class RunSpec:
    config: PilotConfig
    plan_payload: dict | None
    plan_path: str | None
    plan_sha256: str | None
    overrides: dict


def _load_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _apply(value, override):
    return value if override is None else override


def build_run_spec(args) -> RunSpec:
    overrides = {}
    if args.plan is not None:
        plan_path = Path(args.plan)
        payload = _load_json(plan_path)
        plan_sha = _sha256_file(plan_path)
        if payload.get("training", False):
            raise ValueError("The pilot plan must not train")
        source = payload["source_release"]
        roots = payload["roots"]
        engine = payload["engine"]
        reference = payload["reference_evaluation"]
        if roots.get("position_key") != POSITION_KEY:
            raise ValueError("The plan uses an unsupported position key")
        if engine.get("search_state") != "new game for every node":
            raise ValueError("The plan requires a new game state for every node")
        teachers = tuple(payload["teachers"])
        if set(teachers) != set(TEACHER_BY_NAME) or len(teachers) != len(
            TEACHER_BY_NAME
        ):
            raise ValueError("The plan must name exactly the three configured teachers")

        values = {
            "release": source["path"],
            "release_sha256": source["sha256"],
            "train_roots": roots["train"],
            "val_roots": roots["val"],
            "seed": roots["seed"],
            "engine": engine["executable"],
            "engine_version": engine.get("version"),
            "depth": engine["depth"],
            "multipv": engine["multipv"],
            "threads": engine["threads"],
            "hash_mb": engine["hash_mb"],
            "max_time": engine.get("max_time"),
            "workers": engine["workers"],
            "teachers": teachers,
            "reference_depth": reference["depth"],
            "reference_val_rows": reference["val_rows"],
        }
        overrides = {
            "train_roots": args.train_roots,
            "val_roots": args.val_roots,
            "seed": args.seed,
            "engine": args.engine,
            "depth": args.depth,
            "multipv": args.multipv,
            "threads": args.threads,
            "hash_mb": args.hash_mb,
            "workers": args.workers,
            "reference_depth": args.reference_depth,
        }
        applied = {
            name: value for name, value in overrides.items() if value is not None
        }
        for name, value in applied.items():
            values[name] = value
        plan_values = {
            "train_roots": roots["train"],
            "val_roots": roots["val"],
            "seed": roots["seed"],
            "engine": engine["executable"],
            "depth": engine["depth"],
            "multipv": engine["multipv"],
            "threads": engine["threads"],
            "hash_mb": engine["hash_mb"],
            "workers": engine["workers"],
            "reference_depth": reference["depth"],
        }
        recorded_overrides = {
            name: value
            for name, value in applied.items()
            if plan_values.get(name) != value
        }
        config = PilotConfig(**values)
        return RunSpec(
            config=config,
            plan_payload=payload,
            plan_path=str(plan_path),
            plan_sha256=plan_sha,
            overrides=recorded_overrides,
        )

    if args.release is None:
        raise ValueError("Give --plan or --release")
    release_path = Path(args.release)
    expected = (
        args.release_sha256
        if args.release_sha256 is not None
        else _sha256_file(release_path)
    )
    values = {
        "release": str(release_path),
        "release_sha256": expected,
        "train_roots": _apply(1024, args.train_roots),
        "val_roots": _apply(512, args.val_roots),
        "seed": _apply(640, args.seed),
        "engine": _apply("stockfish", args.engine),
        "engine_version": None,
        "depth": _apply(8, args.depth),
        "multipv": _apply(4, args.multipv),
        "threads": _apply(1, args.threads),
        "hash_mb": _apply(64, args.hash_mb),
        "max_time": None,
        "workers": _apply(8, args.workers),
        "teachers": tuple(spec.name for spec in TEACHER_SPECS),
        "reference_depth": _apply(16, args.reference_depth),
        "reference_val_rows": 512,
    }
    return RunSpec(
        config=PilotConfig(**values),
        plan_payload=None,
        plan_path=None,
        plan_sha256=None,
        overrides={},
    )


def _config_payload(run_spec: RunSpec) -> dict:
    return {
        "format_version": 1,
        "kind": "sequence-pilot-config",
        "config": run_spec.config.as_dict(),
        "plan": (
            None
            if run_spec.plan_payload is None
            else {
                "path": run_spec.plan_path,
                "sha256": run_spec.plan_sha256,
                "payload": run_spec.plan_payload,
            }
        ),
        "overrides": run_spec.overrides,
    }


def ensure_config(output: Path, run_spec: RunSpec) -> dict:
    payload = _config_payload(run_spec)
    path = Path(output) / "config.json"
    if path.exists():
        existing = load_manifest(path)
        if _canonical(existing) != _canonical(payload):
            raise ValueError(
                "The output directory belongs to a different configuration"
            )
        return existing
    write_manifest(path, payload)
    return payload


def _resolve_release(config: PilotConfig) -> Path:
    path = Path(config.release)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def load_engine_identity(config: PilotConfig) -> dict:
    executable = shutil.which(config.engine) or config.engine
    path = Path(executable).resolve()
    if not path.is_file():
        raise ValueError(f"Stockfish executable not found: {config.engine}")
    with chess.engine.SimpleEngine.popen_uci(str(path)) as engine:
        identity = dict(engine.id)
        defaults = {name: option.default for name, option in engine.options.items()}
    if config.engine_version and config.engine_version not in identity.get("name", ""):
        raise ValueError(
            f"Engine identity {identity.get('name')!r} does not contain "
            f"the planned version {config.engine_version!r}"
        )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "id": identity,
        "defaults": defaults,
        "expected_version": config.engine_version,
    }


class JsonlStore:
    """Append-only identity-checked JSONL with tail recovery on load."""

    def __init__(self, path, identity_sha256: str, key: str = "root_id"):
        self.path = Path(path)
        self.identity_sha256 = identity_sha256
        self.key = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def _truncate(self, offset: int):
        with self.path.open("r+b") as handle:
            handle.truncate(offset)
            handle.flush()
            os.fsync(handle.fileno())

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        raw = self.path.read_bytes()
        complete = raw.rfind(b"\n") + 1
        if complete < len(raw):
            self._truncate(complete)
        body = raw[:complete]
        lines = body.split(b"\n")
        if lines and lines[-1] == b"":
            lines.pop()
        records = {}
        offset = 0
        for index, line in enumerate(lines):
            start = offset
            offset += len(line) + 1
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                if index == len(lines) - 1:
                    self._truncate(start)
                    break
                raise ValueError(
                    f"Malformed complete record in {self.path} at line {index + 1}"
                )
            if record.get("identity_sha256") != self.identity_sha256:
                raise ValueError(
                    f"Record identity differs from the current identity in {self.path}"
                )
            key = record.get(self.key)
            if key is None:
                raise ValueError(f"A record in {self.path} has no {self.key}")
            if key in records:
                raise ValueError(f"Duplicate {self.key} {key!r} in {self.path}")
            records[key] = record
        return records

    def append(self, record: dict):
        if record.get("identity_sha256") != self.identity_sha256:
            raise ValueError("Refusing to append a record with a different identity")
        line = (
            json.dumps(record, separators=(",", ":"), ensure_ascii=True) + "\n"
        ).encode("utf-8")
        with self.path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def _append_jsonl(path: Path, records) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        for record in records:
            handle.write(
                (
                    json.dumps(record, separators=(",", ":"), ensure_ascii=True) + "\n"
                ).encode("utf-8")
            )
        handle.flush()
        os.fsync(handle.fileno())


def _repair_jsonl_tail(path: Path) -> None:
    """Drop an unterminated final line before appending after a crash."""
    path = Path(path)
    if not path.exists():
        return
    raw = path.read_bytes()
    complete = raw.rfind(b"\n") + 1
    if complete < len(raw):
        with path.open("r+b") as handle:
            handle.truncate(complete)
            handle.flush()
            os.fsync(handle.fileno())


def _write_jsonl_atomic(path: Path, records) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            for record in records:
                handle.write(
                    (
                        json.dumps(record, separators=(",", ":"), ensure_ascii=True)
                        + "\n"
                    ).encode("utf-8")
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Missing JSONL file: {path}")
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"Unterminated final record in {path}")
    return [json.loads(line.decode("utf-8")) for line in raw.splitlines() if line]


def _read_optional_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    complete = raw.rfind(b"\n") + 1
    records = []
    for line in raw[:complete].splitlines():
        if not line:
            continue
        try:
            records.append(json.loads(line.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return records


def _root_from_row(
    row,
    split,
    shard_index,
    local_row,
    global_row,
    shard,
    shard_path,
    release_path,
    release_sha256,
):
    if not isinstance(row, dict):
        return None, "invalid_row"
    if row.get("terminal"):
        return None, "terminal"
    fen = row.get("fen")
    if not isinstance(fen, str):
        return None, "invalid_row"
    fields = fen.split()
    if len(fields) < 4 or fields[1] not in {"w", "b"}:
        return None, "invalid_row"
    repetition = row.get("repetition", 1)
    if (
        isinstance(repetition, bool)
        or not isinstance(repetition, int)
        or repetition < 1
    ):
        return None, "invalid_row"
    if repetition >= 5:
        return None, "terminal"
    unit = row.get("unit")
    d16 = None
    if isinstance(unit, dict):
        for entry in unit.get("per_depth", []):
            if isinstance(entry, dict) and entry.get("d") == 16:
                d16 = entry.get("r1")
                break
    if (
        not isinstance(d16, dict)
        or not isinstance(d16.get("bm"), str)
        or "value" not in d16
        or "mate" not in d16
    ):
        return None, "incomplete_reference"
    value = d16["value"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not -1 <= float(value) <= 1
    ):
        return None, "incomplete_reference"
    if d16.get("mate") is not None and (
        isinstance(d16["mate"], bool) or not isinstance(d16["mate"], int)
    ):
        return None, "incomplete_reference"
    key = " ".join(fields[:4])
    root_id = _sha256_bytes(_canonical([key, repetition]).encode("utf-8"))
    return {
        "root_id": root_id,
        "split": split,
        "fen": fen,
        "repetition": repetition,
        "turn": "white" if fields[1] == "w" else "black",
        "position_key": key,
        "d16": d16,
        "source": {
            "release": str(release_path),
            "release_sha256": release_sha256,
            "split": split,
            "shard": str(shard_path),
            "shard_index": shard_index,
            "shard_sha256": shard["sha256"],
            "row": local_row,
            "global_row": global_row,
            "annotation_position_id": row.get("position_id"),
            "annotation_source_row": row.get("source_row"),
            "annotation_root_player": row.get("root_player"),
        },
    }, None


def _scan_split(
    split,
    shards,
    seed,
    count,
    release_path,
    release_sha256,
    forbidden,
):
    rng = random.Random(seed)
    reservoir = []
    seen = set()
    stats = Counter()
    total = 0
    eligible = 0
    for shard_index, shard in enumerate(shards):
        shard_path = Path(shard["path"])
        if not shard_path.is_absolute():
            shard_path = Path(release_path).parent / shard_path
        expected_rows = shard["rows"]
        digest = sha256()
        rows_here = 0
        with shard_path.open("rb") as handle:
            for local_row, line in enumerate(handle):
                digest.update(line)
                global_row = total
                total += 1
                rows_here += 1
                try:
                    row = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"Malformed release row {shard_path}:{local_row}"
                    ) from error
                root, reason = _root_from_row(
                    row,
                    split,
                    shard_index,
                    local_row,
                    global_row,
                    shard,
                    shard_path,
                    release_path,
                    release_sha256,
                )
                if reason is not None:
                    stats[reason] += 1
                    continue
                key = root["position_key"]
                if forbidden is not None and key in forbidden:
                    stats["split_overlap"] += 1
                    continue
                if key in seen:
                    stats["duplicate_position"] += 1
                    continue
                seen.add(key)
                eligible += 1
                if len(reservoir) < count:
                    reservoir.append(root)
                else:
                    replacement = rng.randrange(eligible)
                    if replacement < count:
                        reservoir[replacement] = root
        if rows_here != expected_rows:
            raise ValueError(
                f"Shard row count mismatch: {shard_path} has {rows_here}, "
                f"manifest says {expected_rows}"
            )
        if digest.hexdigest() != shard["sha256"]:
            raise ValueError(f"Source checksum mismatch: {shard_path}")
    stats["rows"] = total
    stats["eligible"] = eligible
    stats["selected"] = len(reservoir)
    if eligible < count:
        raise ValueError(
            f"Split {split} has {eligible} eligible roots; {count} were requested"
        )
    return reservoir, stats


def _public_stats(stats):
    return {
        "rows": stats["rows"],
        "eligible": stats["eligible"],
        "selected": stats["selected"],
        "excluded": {
            key: value
            for key, value in stats.items()
            if key not in {"rows", "eligible", "selected"}
        },
    }


def _validate_roots(roots):
    ids = set()
    keys = set()
    split_keys = {"train": set(), "val": set()}
    for root in roots:
        root_id = root["root_id"]
        if root_id in ids:
            raise ValueError(f"Duplicate root id: {root_id}")
        ids.add(root_id)
        key = root["position_key"]
        if key != " ".join(root["fen"].split()[:4]):
            raise ValueError(f"Root position key disagrees with its FEN: {root_id}")
        if key in keys:
            raise ValueError(f"Duplicate root position: {key}")
        keys.add(key)
        split_keys[root["split"]].add(key)
        board = chess.Board(root["fen"])
        if board.is_game_over():
            raise ValueError(f"Selected root is terminal: {root_id}")
        if board.turn != (root["turn"] == "white"):
            raise ValueError(f"Selected root turn disagrees with its FEN: {root_id}")
        if root["repetition"] < 1 or root["repetition"] >= 5:
            raise ValueError(f"Selected root repetition is invalid: {root_id}")
        encode_board(root["fen"], root["repetition"])
        d16 = root["d16"]
        move = chess.Move.from_uci(d16["bm"])
        if move not in board.legal_moves:
            raise ValueError(f"Selected root d16 move is illegal: {root_id}")
        if not -1 <= float(d16["value"]) <= 1:
            raise ValueError(f"Selected root d16 value is invalid: {root_id}")
    overlap = split_keys["train"] & split_keys["val"]
    if overlap:
        raise ValueError(f"Train/validation roots overlap: {len(overlap)} positions")


def select_roots(config: PilotConfig, output: Path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    release_path = _resolve_release(config)
    actual_sha = _sha256_file(release_path)
    if actual_sha != config.release_sha256:
        raise ValueError(
            f"Release checksum mismatch: {release_path} is {actual_sha}, "
            f"expected {config.release_sha256}"
        )
    release = load_manifest(release_path)
    if release.get("format_version") != 2:
        raise ValueError("Root selection needs a version-2 annotation release")
    files = release.get("files")
    if not isinstance(files, dict) or not {"train", "val"}.issubset(files):
        raise ValueError("The release must have train and val shards")
    for split in ("train", "val"):
        entries = files[split]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"The release has no {split} shards")
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("path"), str)
                or not isinstance(entry.get("sha256"), str)
                or not isinstance(entry.get("rows"), int)
                or entry["rows"] < 1
            ):
                raise ValueError(f"Malformed {split} shard entry: {entry!r}")

    val_roots, val_stats = _scan_split(
        "val",
        files["val"],
        config.seed,
        config.val_roots,
        release_path,
        config.release_sha256,
        None,
    )
    val_keys = {root["position_key"] for root in val_roots}
    train_roots, train_stats = _scan_split(
        "train",
        files["train"],
        config.seed,
        config.train_roots,
        release_path,
        config.release_sha256,
        val_keys,
    )
    roots = sorted(train_roots, key=lambda root: root["source"]["global_row"])
    roots += sorted(val_roots, key=lambda root: root["source"]["global_row"])
    _validate_roots(roots)

    first_shard = Path(files["train"][0]["path"])
    if not first_shard.is_absolute():
        first_shard = release_path.parent / first_shard
    source_teacher = {}
    source_manifest_path = first_shard.with_suffix(".manifest.json")
    if source_manifest_path.exists():
        source_teacher = load_manifest(source_manifest_path)
        if source_teacher.get("format_version") != 2:
            raise ValueError("The source shard manifest is not format 2")

    roots_path = output / "roots.jsonl"
    _write_jsonl_atomic(roots_path, roots)
    manifest = {
        "format_version": ROOT_FORMAT_VERSION,
        "kind": "sequence-pilot-roots",
        "release": {
            "path": str(release_path),
            "sha256": config.release_sha256,
            "format_version": release.get("format_version"),
        },
        "selection": {
            "seed": config.seed,
            "train": config.train_roots,
            "val": config.val_roots,
            "position_key": config.position_key,
            "algorithm": ROOT_RNG,
            "scan_order": ["val", "train"],
            "train_filter": "selected validation position keys",
        },
        "counts": {
            "train": _public_stats(train_stats),
            "val": _public_stats(val_stats),
        },
        "source_teacher": source_teacher,
        "files": {
            "roots.jsonl": {
                "sha256": _sha256_file(roots_path),
                "rows": len(roots),
            }
        },
    }
    write_manifest(output / "roots.manifest.json", manifest)
    return roots, manifest


def _roots_manifest_matches(manifest, config: PilotConfig):
    if manifest.get("kind") != "sequence-pilot-roots":
        return False
    if manifest.get("format_version") != ROOT_FORMAT_VERSION:
        return False
    release = manifest.get("release", {})
    if release.get("sha256") != config.release_sha256:
        return False
    selection = manifest.get("selection", {})
    return (
        selection.get("seed") == config.seed
        and selection.get("train") == config.train_roots
        and selection.get("val") == config.val_roots
        and selection.get("position_key") == config.position_key
        and selection.get("algorithm") == ROOT_RNG
    )


def ensure_roots(config: PilotConfig, output: Path):
    output = Path(output)
    roots_path = output / "roots.jsonl"
    manifest_path = output / "roots.manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        if not _roots_manifest_matches(manifest, config):
            raise ValueError(
                "The existing root selection belongs to another configuration"
            )
        if not roots_path.exists():
            raise ValueError("roots.jsonl is missing beside its manifest")
        expected = manifest["files"]["roots.jsonl"]
        if _sha256_file(roots_path) != expected["sha256"]:
            raise ValueError("roots.jsonl does not match its manifest checksum")
        roots = _read_jsonl(roots_path)
        if len(roots) != expected["rows"]:
            raise ValueError("roots.jsonl row count does not match its manifest")
        _validate_roots(roots)
        return roots, manifest
    if roots_path.exists() and (
        (output / "synthesis.jsonl").exists()
        or (output / "synthesis.identity.json").exists()
        or (output / "synthesis.manifest.json").exists()
    ):
        raise ValueError(
            "roots.jsonl exists without a manifest; refusing to replace data "
            "beside synthesis artifacts"
        )
    return select_roots(config, output)


def synthesis_identity_payload(
    config: PilotConfig,
    roots_path: Path,
    roots_manifest_path: Path,
    engine_identity: dict,
):
    roots_sha = _sha256_file(roots_path)
    roots_manifest_sha = _sha256_file(roots_manifest_path)
    roots_manifest = load_manifest(roots_manifest_path)
    source_engine_sha = (roots_manifest.get("source_teacher") or {}).get(
        "engine_sha256"
    )
    identity = {
        "format_version": SYNTHESIS_FORMAT_VERSION,
        "kind": "teacher-sequence-synthesis",
        "release": {
            "path": str(_resolve_release(config)),
            "sha256": config.release_sha256,
        },
        "roots": {
            "sha256": roots_sha,
            "manifest_sha256": roots_manifest_sha,
            "train": config.train_roots,
            "val": config.val_roots,
            "seed": config.seed,
            "position_key": config.position_key,
        },
        "engine": engine_identity,
        "source": {
            "source_engine_sha256": source_engine_sha,
            "source_engine_matches": source_engine_sha == engine_identity.get("sha256"),
        },
        "analysis": {
            "engine": engine_identity["path"],
            "depth": config.depth,
            "multipv": config.multipv,
            "threads": config.threads,
            "hash_mb": config.hash_mb,
            "max_time": config.max_time,
            "required_depths": list(REQUIRED_DEPTHS),
            "score_perspective": "root_player",
            "annotation_perspective": "node side to move",
            "value_scale": "(wdl_wins - wdl_losses) / wdl_total",
            "uci_show_wdl": True,
            "search_state": "new game for every node",
        },
        "teachers": {name: TEACHER_BY_NAME[name].as_dict() for name in config.teachers},
        "semantics": {
            "node_identity": "absolute UCI path from the root",
            "root_perspective": "odd path depth negates values and swaps mate sides",
            "parent_aggregation": (
                "visible quantized value and mate; max when the parent side to move "
                "equals the root player, otherwise min; only complete child sets"
            ),
            "leaf_value": "d1 and d8 baselines at every non-terminal node in the root frame",
            "terminal_value": "checkmate mate 0 or draw value 0 in the root frame",
            "root_final": (
                "root leaf uses d8 rank 1; an expanded root shallow-screens every "
                "root candidate, deepens unstable ones in complete rounds at target "
                "depths two and three, and decides from every candidate's latest "
                "visible value and mate, first proposal on exact ties"
            ),
            "trace_grammar": TRACE_GRAMMAR_VERSION,
            "tokenizer": CHESS_TOKENIZER_VERSION,
        },
        "board": {
            "root_prefix_always_observed": True,
            "return_plies": BOARD_RETURN_PLIES,
            "unobserved_plies": BOARD_PLY_INTERVAL,
            "revisit_is_location_not_expansion": True,
            "revisit_restarts_departure_clock": True,
            "no_repeat_without_new_move": True,
        },
        "truncation": {
            "zero_thought": "full teacher final move",
            "positive_prefix": "state-derived current best move",
            "board_response_atomic": True,
        },
        "constants": {
            "stability_value_floor": STABILITY_VALUE_FLOOR,
            "stability_near_value": STABILITY_NEAR_VALUE,
            "max_adaptive_candidates": MAX_ADAPTIVE_CANDIDATES,
            "value_epsilon": VALUE_EPSILON,
            "board_return_plies": BOARD_RETURN_PLIES,
            "board_ply_interval": BOARD_PLY_INTERVAL,
        },
    }
    return {
        "format_version": 1,
        "kind": "sequence-pilot-synthesis-identity",
        "sha256": _identity_sha(identity),
        "identity": identity,
    }


def ensure_identity(path: Path, payload: dict, kind: str) -> dict:
    path = Path(path)
    if path.exists():
        existing = load_manifest(path)
        if _canonical(existing) != _canonical(payload):
            raise ValueError(f"The existing {kind} identity differs from this run")
        return existing
    write_manifest(path, payload)
    return payload


def load_identity(path: Path, kind: str) -> dict:
    payload = load_manifest(path)
    if payload.get("kind") != kind:
        raise ValueError(f"Not a {kind} identity file: {path}")
    if _identity_sha(payload["identity"]) != payload.get("sha256"):
        raise ValueError(f"Identity checksum mismatch: {path}")
    return payload


def _annotation_config(config: PilotConfig, engine_path: str) -> AnnotationConfig:
    return AnnotationConfig(
        engine=engine_path,
        depth=config.depth,
        multipv=config.multipv,
        threads=config.threads,
        hash_mb=config.hash_mb,
        max_time=config.max_time,
    )


class _WorkerEngines:
    """Lazily open one configured Stockfish process per worker thread."""

    def __init__(self, engine_path: str, threads: int, hash_mb: int):
        self.engine_path = engine_path
        self.threads = threads
        self.hash_mb = hash_mb
        self.local = threading.local()
        self.engines = []

    def get(self):
        engine = getattr(self.local, "engine", None)
        if engine is None:
            engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            engine.configure({"Threads": self.threads, "Hash": self.hash_mb})
            self.local.engine = engine
            self.engines.append(engine)
        return engine

    def discard(self, engine):
        try:
            engine.quit()
        except Exception:
            pass
        if engine in self.engines:
            self.engines.remove(engine)
        if getattr(self.local, "engine", None) is engine:
            self.local.engine = None

    def close(self):
        for engine in self.engines:
            try:
                engine.quit()
            except Exception:
                pass
        self.engines.clear()


def _run_parallel(pending, workers, task, on_success, on_failure):
    if not pending:
        return 0, 0
    written = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
        futures = deque()
        iterator = iter(pending)

        def finish(future, root):
            nonlocal written, failed
            try:
                result = future.result()
            except Exception as error:
                on_failure(root, error)
                failed += 1
            else:
                on_success(result)
                written += 1

        for _ in range(2 * min(workers, len(pending))):
            root = next(iterator, None)
            if root is None:
                break
            futures.append((pool.submit(task, root), root))
        while futures:
            future, root = futures.popleft()
            finish(future, root)
            following = next(iterator, None)
            if following is not None:
                futures.append((pool.submit(task, following), following))
    return written, failed


def run_synthesis(
    config: PilotConfig,
    roots,
    output: Path,
    identity_sha: str,
    engine_identity: dict,
    workers: int,
):
    output = Path(output)
    store = JsonlStore(output / "synthesis.jsonl", identity_sha)
    failures_path = output / "synthesis.failures.jsonl"
    failures_path.touch(exist_ok=True)
    _repair_jsonl_tail(failures_path)
    done = set(store.load())
    pending = [root for root in roots if root["root_id"] not in done]
    annotation_config = _annotation_config(config, engine_identity["path"])
    specs = tuple(TEACHER_BY_NAME[name] for name in config.teachers)
    engines = _WorkerEngines(engine_identity["path"], config.threads, config.hash_mb)

    def task(root):
        engine = engines.get()
        try:
            record = synthesize_root(engine, root, annotation_config, specs)
        except Exception:
            engines.discard(engine)
            raise
        record["identity_sha256"] = identity_sha
        return record

    def success(record):
        store.append(record)
        done.add(record["root_id"])

    def failure(root, error):
        _append_jsonl(
            failures_path,
            [
                {
                    "identity_sha256": identity_sha,
                    "root_id": root["root_id"],
                    "split": root["split"],
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc()[-8000:],
                    "time": time.time(),
                }
            ],
        )

    start = time.monotonic()
    try:
        written, failed = _run_parallel(pending, workers, task, success, failure)
    finally:
        engines.close()
    elapsed_ms = int((time.monotonic() - start) * 1000)
    missing = [root for root in roots if root["root_id"] not in done]
    manifest = {
        "format_version": SYNTHESIS_FORMAT_VERSION,
        "kind": "sequence-pilot-synthesis",
        "identity_sha256": identity_sha,
        "counts": {
            "requested": len(roots),
            "written": len(done),
            "missing": len(missing),
            "written_this_run": written,
            "failed_this_run": failed,
        },
        "files": {
            "synthesis.jsonl": {
                "sha256": _sha256_file(store.path),
                "rows": len(done),
            },
            "synthesis.identity.json": _sha256_file(output / "synthesis.identity.json"),
            "synthesis.failures.jsonl": _sha256_file(failures_path),
        },
        "elapsed_this_run_ms": elapsed_ms,
    }
    write_manifest(output / "synthesis.manifest.json", manifest)
    return {
        "written": len(done),
        "missing": len(missing),
        "written_this_run": written,
        "failed_this_run": failed,
        "elapsed_ms": elapsed_ms,
    }


def _reference_val_roots(config, roots, synthesis):
    """Validation roots with synthesis records, truncated to the reference budget.

    This is the single selection used by both the quality run and the summary.
    Returns the requested roots and the number of validation roots available
    before truncation.
    """
    available = [
        root
        for root in roots
        if root["split"] == "val" and root["root_id"] in synthesis
    ]
    return available[: config.reference_val_rows], len(available)


def quality_identity_payload(config, roots, synthesis, synthesis_sha, engine_identity):
    requested, available = _reference_val_roots(config, roots, synthesis)
    identity = {
        "format_version": QUALITY_FORMAT_VERSION,
        "kind": "sequence-pilot-quality",
        "synthesis_identity_sha256": synthesis_sha,
        "reference": {
            "depth": config.reference_depth,
            "val_rows": len(requested),
            "available_val_rows": available,
            "planned_val_rows": config.reference_val_rows,
            "method": (
                "depth-limited search with multipv=1 and root_moves set to the "
                "teacher final root move; value from WDL in the root player frame"
            ),
            "score_perspective": "root_player",
        },
        "engine": engine_identity,
    }
    return {
        "format_version": 1,
        "kind": "sequence-pilot-quality-identity",
        "sha256": _identity_sha(identity),
        "identity": identity,
    }


def _synthesis_identity_sha(output: Path) -> str:
    identity_path = Path(output) / "synthesis.identity.json"
    if identity_path.exists():
        return load_identity(identity_path, "sequence-pilot-synthesis-identity")[
            "sha256"
        ]
    manifest_path = Path(output) / "synthesis.manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        if manifest.get("kind") == "sequence-pilot-synthesis":
            return manifest["identity_sha256"]
    raise ValueError("No synthesis identity found")


def run_quality(
    config: PilotConfig,
    roots,
    output: Path,
    synthesis_sha: str,
    engine_identity: dict,
    workers: int,
):
    output = Path(output)
    synth_store = JsonlStore(output / "synthesis.jsonl", synthesis_sha)
    synthesis = synth_store.load()
    val_roots, _ = _reference_val_roots(config, roots, synthesis)
    quality_sha = load_identity(
        output / "quality.identity.json", "sequence-pilot-quality-identity"
    )["sha256"]
    store = JsonlStore(output / "quality.jsonl", quality_sha)
    failures_path = output / "quality.failures.jsonl"
    failures_path.touch(exist_ok=True)
    _repair_jsonl_tail(failures_path)
    done = store.load()
    pending = [root for root in val_roots if root["root_id"] not in done]
    engines = _WorkerEngines(engine_identity["path"], config.threads, config.hash_mb)

    def task(root):
        record = synthesis[root["root_id"]]
        board = chess.Board(record["fen"])
        cache = {}
        teachers = {}
        engine = engines.get()
        try:
            for name in config.teachers:
                teacher = record["teachers"].get(name)
                if teacher is None:
                    continue
                move = teacher["final_move"]
                if move not in cache:
                    cache[move] = restricted_value(
                        engine, board, move, config.reference_depth
                    )
                restricted = cache[move]
                reference = record["d16"]
                teachers[name] = {
                    "final_move": move,
                    "top1": move == reference["bm"],
                    "restricted": restricted,
                    "reference": {
                        "move": reference["bm"],
                        "value": float(reference["value"]),
                        "mate": reference.get("mate"),
                        "cp": reference.get("cp"),
                    },
                    "delta": restricted["value"] - float(reference["value"]),
                }
        except Exception:
            engines.discard(engine)
            raise
        return {
            "identity_sha256": quality_sha,
            "root_id": root["root_id"],
            "split": "val",
            "teachers": teachers,
            "unique_moves": len(cache),
        }

    def success(record):
        store.append(record)
        done[record["root_id"]] = record

    def failure(root, error):
        _append_jsonl(
            failures_path,
            [
                {
                    "identity_sha256": quality_sha,
                    "root_id": root["root_id"],
                    "split": root["split"],
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc()[-8000:],
                    "time": time.time(),
                }
            ],
        )

    start = time.monotonic()
    try:
        written, failed = _run_parallel(pending, workers, task, success, failure)
    finally:
        engines.close()
    missing = [root for root in val_roots if root["root_id"] not in done]
    manifest = {
        "format_version": QUALITY_FORMAT_VERSION,
        "kind": "sequence-pilot-quality",
        "identity_sha256": quality_sha,
        "counts": {
            "requested": len(val_roots),
            "written": len(done),
            "missing": len(missing),
            "written_this_run": written,
            "failed_this_run": failed,
        },
        "files": {
            "quality.jsonl": {
                "sha256": _sha256_file(store.path),
                "rows": len(done),
            },
            "quality.identity.json": _sha256_file(output / "quality.identity.json"),
            "quality.failures.jsonl": _sha256_file(failures_path),
        },
        "elapsed_this_run_ms": int((time.monotonic() - start) * 1000),
    }
    write_manifest(output / "quality.manifest.json", manifest)
    return {
        "written": len(done),
        "missing": len(missing),
        "written_this_run": written,
        "failed_this_run": failed,
    }


def validate_synthesis(config: PilotConfig, roots, output: Path):
    output = Path(output)
    synthesis_sha = _synthesis_identity_sha(output)
    store = JsonlStore(output / "synthesis.jsonl", synthesis_sha)
    records = store.load()
    root_by_id = {root["root_id"]: root for root in roots}
    teacher_count = 0
    state_count = 0
    replayed = 0
    for root_id, record in records.items():
        root = root_by_id.get(root_id)
        if root is None:
            raise ValueError(f"Synthesis record has no selected root: {root_id}")
        if (
            record["fen"] != root["fen"]
            or record["repetition"] != root["repetition"]
            or record["d16"] != root["d16"]
        ):
            raise ValueError(f"Synthesis root provenance changed: {root_id}")
        board = chess.Board(record["fen"])
        if board.turn != (record["turn"] == "white"):
            raise ValueError(f"Synthesis root turn is invalid: {root_id}")
        for name in config.teachers:
            teacher = record["teachers"].get(name)
            if teacher is None:
                raise ValueError(f"Synthesis record lacks {name}: {root_id}")
            operations = teacher["operations"]
            planned = plan_boards(record["fen"], record["repetition"], operations)
            if list(planned) != teacher["board_slots"]:
                raise ValueError(f"BOARD plan changed for {root_id}/{name}")
            no_tool = emit_operations(
                record["fen"],
                record["repetition"],
                operations,
                planned,
                tool=False,
            )
            tool = emit_operations(
                record["fen"],
                record["repetition"],
                operations,
                planned,
                tool=True,
            )
            if list(no_tool.tokens) != teacher["no_tool"]["tokens"]:
                raise ValueError(f"No-tool tokens changed for {root_id}/{name}")
            if list(tool.tokens) != teacher["tool"]["tokens"]:
                raise ValueError(f"Tool tokens changed for {root_id}/{name}")
            stored_spans = teacher["truncation"]["board_spans"]
            if stored_spans != [[start, end] for _, start, end, _ in tool.board_spans]:
                raise ValueError(f"BOARD spans changed for {root_id}/{name}")
            states = build_states(operations, no_tool, tool)
            if states != teacher["truncation"]["states"]:
                raise ValueError(f"Truncation states changed for {root_id}/{name}")
            verification = verify_teacher_trace(
                record["fen"],
                record["repetition"],
                operations,
                record["nodes"],
                no_tool,
                tool,
                states,
                TEACHER_BY_NAME[name],
            )
            for field in (
                "operations",
                "tool_steps",
                "board_steps",
                "states",
                "materialized_states",
            ):
                if verification[field] != teacher["verification"].get(field):
                    raise ValueError(
                        f"Verification {field} disagrees for {root_id}/{name}"
                    )
            if teacher["final_move"] != operations[-1]["uci"]:
                raise ValueError(f"Final move disagrees for {root_id}/{name}")
            teacher_count += 1
            state_count += len(states)
            replayed += 1
    result = {
        "format_version": 1,
        "kind": "sequence-pilot-validation",
        "identity_sha256": synthesis_sha,
        "records": len(records),
        "teachers": teacher_count,
        "replayed": replayed,
        "states": state_count,
        "ok": True,
    }
    write_manifest(output / "validation.json", result)
    return result


def _summary_teacher(config):
    return {
        "roots": 0,
        "nodes_total": 0,
        "nodes_min": None,
        "nodes_max": 0,
        "depth_max": 0,
        "width_max": 0,
        "proposed_total": 0,
        "root_leaf": 0,
        "multi_candidate_roots": 0,
        "overturned_roots": 0,
        "non_root_branch_roots": 0,
        "updates": 0,
        "revisions": 0,
        "leaf_count": 0,
        "leaf_shift": 0,
        "states": 0,
        "state_transitions": 0,
        "state_changes": 0,
        "roots_with_state_changes": 0,
        "stop_depths": Counter(),
        "final_basis": Counter(),
        "board_requests": 0,
        "roots_with_boards": 0,
        "board_response_tokens": 0,
        "no_tool_tokens": 0,
        "tool_tokens": 0,
        "model_tokens": 0,
        "token_increase": 0,
        "engine_calls": 0,
        "analysis_ms": 0,
        "build_ms": 0,
        "max_width_sum": 0,
        "depth_sum": 0,
        "quality_roots": 0,
        "top1": 0,
        "delta_sum": 0.0,
        "abs_delta_sum": 0.0,
        "delta_min": None,
        "delta_max": None,
        "better": 0,
        "equal": 0,
        "worse": 0,
        "reference_sum": 0.0,
        "restricted_sum": 0.0,
    }


def build_summary(
    config,
    roots,
    synthesis,
    quality,
    validation,
    synthesis_failures,
    quality_failures,
):
    teachers = {name: _summary_teacher(config) for name in config.teachers}
    analysis_calls = 0
    analysis_ms = 0
    root_elapsed_ms = 0
    root_nodes = 0
    terminal_nodes = 0
    for record in synthesis.values():
        metrics = record.get("analysis", {})
        analysis_calls += metrics.get("calls", 0)
        analysis_ms += metrics.get("elapsed_ms", 0)
        root_elapsed_ms += record.get("elapsed_ms", 0)
        root_nodes += metrics.get("nodes", 0)
        terminal_nodes += metrics.get("terminal_nodes", 0)
        for name, teacher in record["teachers"].items():
            if name not in teachers:
                continue
            aggregate = teachers[name]
            metrics = teacher["metrics"]
            aggregate["roots"] += 1
            nodes = metrics["nodes"]
            aggregate["nodes_total"] += nodes
            aggregate["nodes_min"] = (
                nodes
                if aggregate["nodes_min"] is None
                else min(aggregate["nodes_min"], nodes)
            )
            aggregate["nodes_max"] = max(aggregate["nodes_max"], nodes)
            aggregate["depth_max"] = max(aggregate["depth_max"], metrics["max_depth"])
            aggregate["width_max"] = max(aggregate["width_max"], metrics["max_width"])
            aggregate["depth_sum"] += metrics["max_depth"]
            aggregate["max_width_sum"] += metrics["max_width"]
            aggregate["proposed_total"] += metrics["proposed"]
            aggregate["root_leaf"] += int(metrics["proposed"] == 0)
            aggregate["multi_candidate_roots"] += int(metrics["proposed"] >= 2)
            aggregate["overturned_roots"] += int(
                metrics["proposed"] >= 2 and metrics["overturned"]
            )
            aggregate["non_root_branch_roots"] += int(
                metrics["non_root_expanded_nodes"] > 0
            )
            aggregate["updates"] += metrics["updates"]
            aggregate["revisions"] += metrics["revisions"]
            aggregate["leaf_count"] += metrics["leaf_count"]
            aggregate["leaf_shift"] += metrics["leaf_shift"]
            aggregate["states"] += metrics["states"]
            aggregate["state_transitions"] += metrics["state_transitions"]
            aggregate["state_changes"] += metrics["state_changes"]
            aggregate["roots_with_state_changes"] += int(metrics["state_changes"] > 0)
            for depth, count in metrics["stop_depths"].items():
                aggregate["stop_depths"][int(depth)] += count
            aggregate["final_basis"][metrics["final_basis"]] += 1
            aggregate["board_requests"] += metrics["board_requests"]
            aggregate["roots_with_boards"] += int(metrics["board_requests"] > 0)
            aggregate["board_response_tokens"] += metrics["board_response_tokens"]
            aggregate["no_tool_tokens"] += metrics["no_tool_tokens"]
            aggregate["tool_tokens"] += metrics["tool_tokens"]
            aggregate["model_tokens"] += metrics["model_tokens"]
            aggregate["token_increase"] += metrics["token_increase"]
            aggregate["engine_calls"] += metrics["engine_calls"]
            aggregate["analysis_ms"] += metrics["analysis_ms"]
            aggregate["build_ms"] += metrics["build_ms"]

    for record in quality.values():
        for name, entry in record["teachers"].items():
            if name not in teachers:
                continue
            aggregate = teachers[name]
            aggregate["quality_roots"] += 1
            aggregate["top1"] += int(entry["top1"])
            delta = entry["delta"]
            aggregate["delta_sum"] += delta
            aggregate["abs_delta_sum"] += abs(delta)
            aggregate["delta_min"] = (
                delta
                if aggregate["delta_min"] is None
                else min(aggregate["delta_min"], delta)
            )
            aggregate["delta_max"] = (
                delta
                if aggregate["delta_max"] is None
                else max(aggregate["delta_max"], delta)
            )
            if delta > 1e-9:
                aggregate["better"] += 1
            elif delta < -1e-9:
                aggregate["worse"] += 1
            else:
                aggregate["equal"] += 1
            aggregate["reference_sum"] += entry["reference"]["value"]
            aggregate["restricted_sum"] += entry["restricted"]["value"]

    teacher_summary = {}
    for name, aggregate in teachers.items():
        roots_count = aggregate["roots"]
        updates = aggregate["updates"]
        transitions = aggregate["state_transitions"]
        no_tool_tokens = aggregate["no_tool_tokens"]
        quality_roots = aggregate["quality_roots"]
        teacher_summary[name] = {
            "roots": roots_count,
            "nodes": {
                "total": aggregate["nodes_total"],
                "mean": _mean(aggregate["nodes_total"], roots_count),
                "min": aggregate["nodes_min"],
                "max": aggregate["nodes_max"],
            },
            "depth": {
                "mean": _mean(aggregate["depth_sum"], roots_count),
                "max": aggregate["depth_max"],
            },
            "width": {
                "mean": _mean(aggregate["max_width_sum"], roots_count),
                "max": aggregate["width_max"],
            },
            "root_candidates": {
                "mean": _mean(aggregate["proposed_total"], roots_count),
            },
            "tokens": {
                "no_tool_total": no_tool_tokens,
                "no_tool_mean": _mean(no_tool_tokens, roots_count),
                "tool_total": aggregate["tool_tokens"],
                "tool_mean": _mean(aggregate["tool_tokens"], roots_count),
                "model_total": aggregate["model_tokens"],
                "model_mean": _mean(aggregate["model_tokens"], roots_count),
            },
            "engine_calls": {
                "total": aggregate["engine_calls"],
                "mean": _mean(aggregate["engine_calls"], roots_count),
            },
            "analysis_s": aggregate["analysis_ms"] / 1000,
            "build_s": aggregate["build_ms"] / 1000,
            "non_root_branch_rate": _rate(
                aggregate["non_root_branch_roots"], roots_count
            ),
            "root_leaf": aggregate["root_leaf"],
            "first_candidate_overturned": {
                "roots_with_multiple": aggregate["multi_candidate_roots"],
                "overturned": aggregate["overturned_roots"],
                "rate": _rate(
                    aggregate["overturned_roots"],
                    aggregate["multi_candidate_roots"],
                ),
            },
            "evaluation_revision": {
                "updates": updates,
                "revisions": aggregate["revisions"],
                "rate": _rate(aggregate["revisions"], updates),
            },
            "leaf_shift": {
                "leaves": aggregate["leaf_count"],
                "shifted": aggregate["leaf_shift"],
                "rate": _rate(aggregate["leaf_shift"], aggregate["leaf_count"]),
            },
            "truncation_change": {
                "states": aggregate["states"],
                "transitions": transitions,
                "changes": aggregate["state_changes"],
                "rate": _rate(aggregate["state_changes"], transitions),
                "roots_with_changes": aggregate["roots_with_state_changes"],
                "change_root_rate": _rate(
                    aggregate["roots_with_state_changes"], roots_count
                ),
            },
            "stop_depths": {
                str(depth): count
                for depth, count in sorted(aggregate["stop_depths"].items())
            },
            "final_basis": dict(aggregate["final_basis"]),
            "board": {
                "requests": aggregate["board_requests"],
                "roots_with_requests": aggregate["roots_with_boards"],
                "request_rate": _rate(aggregate["roots_with_boards"], roots_count),
                "response_tokens": aggregate["board_response_tokens"],
                "response_ratio_no_tool": _rate(
                    aggregate["board_response_tokens"], no_tool_tokens
                ),
                "response_ratio_tool": _rate(
                    aggregate["board_response_tokens"],
                    aggregate["tool_tokens"],
                ),
                "token_increase": aggregate["token_increase"],
                "increase_rate": _rate(aggregate["token_increase"], no_tool_tokens),
            },
            "quality": (
                None
                if not quality_roots
                else {
                    "roots": quality_roots,
                    "top1": aggregate["top1"],
                    "top1_rate": _rate(aggregate["top1"], quality_roots),
                    "mean_delta": _mean(aggregate["delta_sum"], quality_roots),
                    "mean_abs_delta": _mean(aggregate["abs_delta_sum"], quality_roots),
                    "min_delta": aggregate["delta_min"],
                    "max_delta": aggregate["delta_max"],
                    "better": aggregate["better"],
                    "equal": aggregate["equal"],
                    "worse": aggregate["worse"],
                    "mean_reference_value": _mean(
                        aggregate["reference_sum"], quality_roots
                    ),
                    "mean_restricted_value": _mean(
                        aggregate["restricted_sum"], quality_roots
                    ),
                }
            ),
        }

    missing = [root["root_id"] for root in roots if root["root_id"] not in synthesis]
    missing_set = set(missing)
    failure_by_root = {}
    for record in synthesis_failures:
        if record.get("root_id") in missing_set:
            failure_by_root[record["root_id"]] = record.get("error")
    quality_roots, _ = _reference_val_roots(config, roots, synthesis)
    quality_expected = {root["root_id"] for root in quality_roots}
    quality_missing = quality_expected - set(quality)
    quality_failure_by_root = {}
    for record in quality_failures:
        if record.get("root_id") in quality_missing:
            quality_failure_by_root[record["root_id"]] = record.get("error")
    return {
        "format_version": 1,
        "kind": "sequence-pilot-summary",
        "roots": {
            "train": config.train_roots,
            "val": config.val_roots,
            "selected": len(roots),
        },
        "synthesis": {
            "requested": len(roots),
            "written": len(synthesis),
            "missing": len(missing),
            "missing_root_ids": missing,
            "failure_root_ids": sorted(failure_by_root),
            "failure_reasons": sorted(
                str(reason) for reason in failure_by_root.values()
            ),
        },
        "analysis": {
            "calls_total": analysis_calls,
            "ms_total": analysis_ms,
            "mean_nodes": _mean(root_nodes, len(synthesis)),
            "terminal_nodes": terminal_nodes,
            "root_elapsed_ms": root_elapsed_ms,
        },
        "teachers": teacher_summary,
        "quality": {
            "requested": len(quality_expected),
            "written": len(quality),
            "missing": len(quality_missing),
            "failure_root_ids": sorted(quality_failure_by_root),
        },
        "validation": validation,
    }


def write_summary(
    config,
    roots,
    output: Path,
    synthesis,
    quality,
    validation,
):
    output = Path(output)
    summary_path = output / "summary.json"
    synthesis_failures = _read_optional_jsonl(output / "synthesis.failures.jsonl")
    quality_failures = _read_optional_jsonl(output / "quality.failures.jsonl")
    summary = build_summary(
        config,
        roots,
        synthesis,
        quality,
        validation,
        synthesis_failures,
        quality_failures,
    )
    write_manifest(summary_path, summary)
    return summary


def write_run_manifest(output: Path, config_payload: dict):
    output = Path(output)
    names = [
        "config.json",
        "roots.jsonl",
        "roots.manifest.json",
        "synthesis.identity.json",
        "synthesis.jsonl",
        "synthesis.failures.jsonl",
        "synthesis.manifest.json",
        "validation.json",
        "quality.identity.json",
        "quality.jsonl",
        "quality.failures.jsonl",
        "quality.manifest.json",
        "summary.json",
    ]
    files = {
        name: _sha256_file(output / name) for name in names if (output / name).exists()
    }
    roots_manifest = (
        load_manifest(output / "roots.manifest.json")
        if (output / "roots.manifest.json").exists()
        else {}
    )
    synthesis_manifest = (
        load_manifest(output / "synthesis.manifest.json")
        if (output / "synthesis.manifest.json").exists()
        else {}
    )
    quality_manifest = (
        load_manifest(output / "quality.manifest.json")
        if (output / "quality.manifest.json").exists()
        else {}
    )
    payload = {
        "format_version": RUN_FORMAT_VERSION,
        "kind": "sequence-pilot-run",
        "config": config_payload,
        "release": roots_manifest.get("release"),
        "roots": {
            "sha256": files.get("roots.jsonl"),
            "counts": roots_manifest.get("counts"),
        },
        "synthesis": {
            "identity_sha256": synthesis_manifest.get("identity_sha256"),
            "counts": synthesis_manifest.get("counts"),
        },
        "quality": {
            "identity_sha256": quality_manifest.get("identity_sha256"),
            "counts": quality_manifest.get("counts"),
        },
        "files": files,
    }
    write_manifest(output / "run.manifest.json", payload)
    return payload


def _load_quality(config, output):
    identity_path = output / "quality.identity.json"
    if not identity_path.exists():
        return {}
    identity = load_identity(identity_path, "sequence-pilot-quality-identity")
    store = JsonlStore(output / "quality.jsonl", identity["sha256"])
    return store.load()


def _print(message):
    print(message, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, help="frozen pilot plan JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--release-sha256")
    parser.add_argument("--train-roots", type=int)
    parser.add_argument("--val-roots", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--engine")
    parser.add_argument("--depth", type=int)
    parser.add_argument("--multipv", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--hash-mb", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--reference-depth", type=int)
    parser.add_argument(
        "--stage",
        choices=["all", "roots", "synthesis", "validate", "quality", "summary"],
        default="all",
    )
    parser.add_argument("--no-quality", action="store_true")
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    run_spec = build_run_spec(args)
    config_payload = ensure_config(output, run_spec)
    config = run_spec.config

    roots, roots_manifest = ensure_roots(config, output)
    _print(
        f"roots: train {roots_manifest['counts']['train']['selected']}, "
        f"val {roots_manifest['counts']['val']['selected']}"
    )
    if args.stage == "roots":
        write_run_manifest(output, config_payload)
        return

    synthesis_sha = None
    engine_identity = None
    validation = None
    synthesis = None
    if args.stage in {"synthesis", "all", "quality"}:
        engine_identity = load_engine_identity(config)
    if args.stage in {"synthesis", "all"}:
        identity_payload = synthesis_identity_payload(
            config,
            output / "roots.jsonl",
            output / "roots.manifest.json",
            engine_identity,
        )
        identity = ensure_identity(
            output / "synthesis.identity.json",
            identity_payload,
            "sequence-pilot-synthesis-identity",
        )
        synthesis_sha = identity["sha256"]
        result = run_synthesis(
            config,
            roots,
            output,
            synthesis_sha,
            engine_identity,
            config.workers,
        )
        _print(
            f"synthesis: written {result['written']}, "
            f"this run {result['written_this_run']}, "
            f"missing {result['missing']}, failed this run {result['failed_this_run']}"
        )
    if args.stage in {"validate", "all"}:
        validation = validate_synthesis(config, roots, output)
        _print(
            f"validation: {validation['records']} records, "
            f"{validation['teachers']} teacher traces, "
            f"{validation['states']} states"
        )
    if args.stage in {"quality", "all"} and not args.no_quality:
        synthesis_sha = synthesis_sha or _synthesis_identity_sha(output)
        if engine_identity is None:
            engine_identity = load_engine_identity(config)
        if synthesis is None:
            synthesis = JsonlStore(output / "synthesis.jsonl", synthesis_sha).load()
        quality_payload = quality_identity_payload(
            config, roots, synthesis, synthesis_sha, engine_identity
        )
        quality_identity = ensure_identity(
            output / "quality.identity.json",
            quality_payload,
            "sequence-pilot-quality-identity",
        )
        result = run_quality(
            config,
            roots,
            output,
            synthesis_sha,
            engine_identity,
            config.workers,
        )
        _print(
            f"quality: written {result['written']}, "
            f"this run {result['written_this_run']}, "
            f"missing {result['missing']}, failed this run {result['failed_this_run']}"
        )
        _print(f"quality identity: {quality_identity['sha256']}")
    if args.stage in {"summary", "all"}:
        if synthesis is None:
            synthesis = JsonlStore(
                output / "synthesis.jsonl", _synthesis_identity_sha(output)
            ).load()
        quality = _load_quality(config, output)
        if validation is None and (output / "validation.json").exists():
            validation = load_manifest(output / "validation.json")
        write_summary(
            config,
            roots,
            output,
            synthesis,
            quality,
            validation,
        )
        _print(f"summary: {output / 'summary.json'}")
    write_run_manifest(output, config_payload)


if __name__ == "__main__":
    main()
