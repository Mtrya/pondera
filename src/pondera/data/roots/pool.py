"""Deduplicate and publish ordered root pools with bounded range reads."""

import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing, contextmanager
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from pondera.chess import _rules as rules
from pondera.data.dataset import load_manifest, write_manifest

FIVEFOLD = 5


class CorpusError(RuntimeError):
    """A root pool that cannot be reproduced from its inputs."""


def validate_record(record, index=0):
    root_id = record.get("root_id")
    split = record.get("split")
    fen = record.get("fen")
    repetition = record.get("repetition")
    if not isinstance(root_id, str) or not root_id:
        raise CorpusError(f"Root on line {index + 1} has no root_id")
    if split not in {"train", "val"}:
        raise CorpusError(f"Root {root_id} has an unknown split")
    if not isinstance(fen, str) or len(fen.split()) != 6:
        raise CorpusError(f"Root {root_id} has an invalid FEN")
    if (
        not isinstance(repetition, int)
        or isinstance(repetition, bool)
        or not 1 <= repetition < FIVEFOLD
    ):
        raise CorpusError(f"Root {root_id} has an invalid repetition count")
    position_key = _position_key(fen)
    if record.get("position_key", position_key) != position_key:
        raise CorpusError(f"Root {root_id} has an inconsistent position key")
    try:
        board = rules.Board(fen)
    except ValueError as error:
        raise CorpusError(f"Root {root_id} has an invalid FEN") from error
    if not board.is_valid():
        raise CorpusError(f"Root {root_id} has an invalid position")
    if not board.has_legal_moves():
        raise CorpusError(f"Root {root_id} has no legal move")
    turn = "white" if board.turn == rules.WHITE else "black"
    if record.get("turn") not in (None, turn):
        raise CorpusError(f"Root {root_id} has a turn inconsistent with its FEN")
    return {**record, "turn": turn, "position_key": position_key}


def pool_path(path):
    path = Path(path)
    if path.is_dir():
        path = path / (POOL_FILE if (path / POOL_FILE).is_file() else "roots.jsonl")
    return path


def read_roots(path):
    """Read a prepared pool or an external JSONL root list."""
    path = pool_path(path)
    if path.suffix == ".parquet":
        return list(RootPool(path).iter_range())
    roots = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise CorpusError(f"Invalid root JSON on line {index + 1}") from error
            if not isinstance(record, dict):
                raise CorpusError(f"Root record on line {index + 1} is not an object")
            root = validate_record(record, index)
            if root["root_id"] in seen:
                raise CorpusError(f"Duplicate root id: {root['root_id']}")
            seen.add(root["root_id"])
            roots.append(root)
    return roots


def root_id_for(fen, repetition):
    """Return the identity of a root from its full FEN and repetition count."""
    return f"{fen}|{repetition}"


POOL_FILE = "roots.parquet"
MANIFEST_FILE = "pool.manifest.json"
ROW_GROUP_SIZE = 8192
ROOT_SCHEMA = pa.schema(
    [
        ("position_key", pa.string()),
        ("repetition", pa.int8()),
        ("halfmove", pa.int32()),
        ("fullmove", pa.int32()),
        ("source", pa.string()),
        ("game", pa.int32()),
        ("ply", pa.int32()),
        ("puzzle", pa.string()),
        ("split", pa.string()),
    ]
)

_RETAINED = """
    SELECT key, repetition, halfmove, fullmove, 0 AS origin, game, ply, NULL AS puzzle
    FROM elite
    UNION ALL
    SELECT key, 1, halfmove, fullmove, 1, NULL, NULL, puzzle
    FROM puzzles WHERE NOT overlaps_elite
"""


def _position_key(fen):
    return " ".join(fen.split()[:4])


class RootIndex:
    """A disposable SQLite index for ordered source candidates."""

    def __init__(self, connection, seed):
        self.connection = connection
        self.connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA cache_size=-131072;
            PRAGMA temp_store=FILE;
            CREATE TABLE elite (
                key TEXT PRIMARY KEY, repetition INTEGER, halfmove INTEGER,
                fullmove INTEGER, game INTEGER, ply INTEGER
            ) WITHOUT ROWID;
            CREATE TABLE puzzles (
                key TEXT PRIMARY KEY, halfmove INTEGER, fullmove INTEGER,
                puzzle TEXT, overlaps_elite INTEGER
            ) WITHOUT ROWID;
            """
        )
        for domain in ("validation", "pool_order"):
            self.connection.create_function(
                domain, 1, _seeded_rank(seed, domain), deterministic=True
            )
        self.cursor = connection.cursor()

    def add_elite(self, candidates):
        self.cursor.executemany(
            """
            INSERT INTO elite VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                repetition=excluded.repetition, halfmove=excluded.halfmove,
                fullmove=excluded.fullmove, game=excluded.game, ply=excluded.ply
            WHERE excluded.repetition > elite.repetition
            """,
            candidates,
        )

    def count_elite(self):
        return self.cursor.execute("SELECT count(*) FROM elite").fetchone()[0]

    def add_puzzle(self, key, halfmove, fullmove, row):
        overlap = self.cursor.execute(
            """
            INSERT OR IGNORE INTO puzzles VALUES (
                ?, ?, ?, ?, EXISTS (SELECT 1 FROM elite WHERE key=?)
            ) RETURNING overlaps_elite
            """,
            (key, halfmove, fullmove, orjson.dumps(row).decode(), key),
        ).fetchone()
        if overlap is None:
            return "duplicates_within_source"
        return "duplicates_elite" if overlap[0] else "retained"

    def validation(self, count):
        keys = {
            key
            for (key,) in self.connection.execute(
                """
                SELECT key FROM (
                    SELECT key FROM elite
                    UNION ALL
                    SELECT key FROM puzzles WHERE NOT overlaps_elite
                )
                ORDER BY validation(key), key LIMIT ?
                """,
                (count,),
            )
        }
        if len(keys) != count:
            raise CorpusError(
                f"Pool has {len(keys)} roots for {count} validation slots"
            )
        return keys

    def ordered(self):
        return self.connection.execute(
            f"SELECT * FROM ({_RETAINED}) ORDER BY pool_order(key), key"
        )


def _seeded_rank(seed, domain):
    prefix = f"{seed}:{domain}:".encode()

    def rank(key):
        return hashlib.sha256(prefix + key.encode()).digest()[:8]

    return rank


def write_pool(output, index, manifest):
    """Publish the complete Parquet file and its manifest together."""
    output = Path(output)
    reports = manifest["sources"]
    names = {
        int(report["kind"] == "lichess_puzzles"): name
        for name, report in reports.items()
    }
    provenance = {name: report["provenance"] for name, report in reports.items()}
    identity = hashlib.sha256(orjson.dumps(provenance, option=orjson.OPT_SORT_KEYS))
    validation = index.validation(manifest["validation_count"])
    counts = {name: Counter(train=0, val=0) for name in provenance}

    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as scratch:
        stage = Path(scratch) / "pool"
        stage.mkdir()
        with pq.ParquetWriter(
            stage / POOL_FILE,
            ROOT_SCHEMA,
            compression="zstd",
            use_dictionary=["source", "split", "repetition"],
        ) as writer:
            ordered = index.ordered()
            while group := list(islice(ordered, ROW_GROUP_SIZE)):
                columns = [[] for _ in ROOT_SCHEMA]
                for (
                    key,
                    repetition,
                    halfmove,
                    fullmove,
                    origin,
                    game,
                    ply,
                    puzzle,
                ) in group:
                    name = names[origin]
                    split = "val" if key in validation else "train"
                    row = (
                        key,
                        repetition,
                        halfmove,
                        fullmove,
                        name,
                        game,
                        ply,
                        puzzle,
                        split,
                    )
                    identity.update(orjson.dumps(row) + b"\n")
                    counts[name][split] += 1
                    for column, value in zip(columns, row, strict=True):
                        column.append(value)
                writer.write_table(pa.Table.from_arrays(columns, schema=ROOT_SCHEMA))
            pool_id = identity.hexdigest()
            writer.add_key_value_metadata({"pool_id": pool_id})

        totals = {
            split: sum(count[split] for count in counts.values())
            for split in ("train", "val")
        }
        if not sum(totals.values()):
            raise CorpusError("The inputs produced no roots")
        for source, count in counts.items():
            reports[source]["counts"] = dict(count)
        manifest.update(
            kind="root-pool",
            format_version=4,
            pool_id=pool_id,
            row_group_size=ROW_GROUP_SIZE,
            counts=totals,
        )
        write_manifest(stage / MANIFEST_FILE, manifest)
        stage.rename(output)
    return manifest


class RootPool:
    """Open an immutable pool and read half-open ranges in its published order."""

    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_dir():
            self.path /= POOL_FILE
        self.manifest = load_manifest(self.path.with_name(MANIFEST_FILE))
        if self.manifest.get("format_version") != 4:
            raise CorpusError(f"Unsupported root-pool manifest beside {self.path}")
        self.pool_id = self.manifest["pool_id"]
        self.group_size = self.manifest["row_group_size"]
        self.provenance = {
            name: report["provenance"]
            for name, report in self.manifest["sources"].items()
        }
        self.file = pq.ParquetFile(self.path)
        metadata = self.file.metadata
        self.rows = metadata.num_rows
        expected_groups = (self.rows + self.group_size - 1) // self.group_size
        if (
            (metadata.metadata or {}).get(b"pool_id") != self.pool_id.encode()
            or self.file.schema_arrow.remove_metadata() != ROOT_SCHEMA
            or self.rows != sum(self.manifest["counts"].values())
            or metadata.num_row_groups != expected_groups
        ):
            raise CorpusError(
                f"Parquet data differs from the root-pool manifest: {self.path}"
            )

    def __len__(self):
        return self.rows

    def iter_range(self, start=0, stop=None):
        """Yield roots at ordinals in [start, stop), decoding one group at a time."""
        stop = self.rows if stop is None else stop
        if not 0 <= start <= stop <= self.rows:
            raise ValueError(
                f"Root range [{start}, {stop}) is outside [0, {self.rows})"
            )
        if start == stop:
            return
        size = self.group_size
        for index in range(start // size, (stop - 1) // size + 1):
            base = index * size
            first = max(0, start - base)
            last = min(size, stop - base)
            group = self.file.read_row_group(index, use_threads=False)
            for row in group.slice(first, last - first).to_pylist():
                key = row["position_key"]
                fen = f"{key} {row['halfmove']} {row['fullmove']}"
                repetition = row["repetition"]
                source = dict(self.provenance[row["source"]])
                if row["puzzle"] is not None:
                    source["puzzle"] = orjson.loads(row["puzzle"])
                else:
                    source.update(game=row["game"], ply=row["ply"])
                yield {
                    "root_id": root_id_for(fen, repetition),
                    "split": row["split"],
                    "fen": fen,
                    "repetition": repetition,
                    "turn": "white" if key.split()[1] == "w" else "black",
                    "position_key": key,
                    "source": source,
                }


@contextmanager
def index_pool(seed):
    """Keep only the bounded SQLite page cache resident during preparation."""
    with TemporaryDirectory(prefix="pondera-roots-") as scratch:
        with closing(sqlite3.connect(Path(scratch) / "roots.sqlite")) as connection:
            yield RootIndex(connection, seed)
