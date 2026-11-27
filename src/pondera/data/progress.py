"""Durable curation progress and ordered publication of a complete release."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

PROGRESS_FILE = ".curation.sqlite"


class ReleaseProgress:
    """One registry for disjoint curation jobs on the same output directory."""

    def __init__(self, output, metadata, *, roots, shard_size, pool_id):
        self.directory = Path(output)
        self.release = self.directory / "release.json"
        self.roots = roots
        self.shard_size = shard_size
        self.shards = (roots + shard_size - 1) // shard_size
        self.metadata = metadata
        self.layout = {"pool_id": pool_id, "shard_size": shard_size}
        identity = json.dumps({**self.layout, "metadata": metadata}, sort_keys=True)
        path = self.directory / PROGRESS_FILE
        if self.release.exists() and not path.exists():
            raise ValueError(
                f"Release was published without a progress registry: {self.release}"
            )
        self.db = sqlite3.connect(path, timeout=120)
        try:
            with self._transaction():
                self.db.execute(
                    "CREATE TABLE IF NOT EXISTS release "
                    "(id INTEGER PRIMARY KEY, identity TEXT, dataset_id TEXT, "
                    "done INTEGER, rows INTEGER, drops INTEGER)"
                )
                self.db.execute(
                    "CREATE TABLE IF NOT EXISTS shard "
                    "(ordinal INTEGER PRIMARY KEY, rows INTEGER, dropped TEXT)"
                )
                stored = self.db.execute(
                    "SELECT identity, dataset_id FROM release WHERE id = 1"
                ).fetchone()
                if stored is None:
                    self.dataset_id = uuid4().hex
                    self.db.execute(
                        "INSERT INTO release VALUES (1, ?, ?, 0, 0, 0)",
                        (identity, self.dataset_id),
                    )
                else:
                    previous, self.dataset_id = stored
                    if previous != identity:
                        if json.loads(previous)["shard_size"] != shard_size:
                            raise ValueError(
                                "The existing release has a different shard layout"
                            )
                        raise ValueError(
                            "The existing release belongs to a different pool or configuration"
                        )
        except BaseException:
            self.db.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.db.close()

    @contextmanager
    def _transaction(self):
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            yield

    def bounds(self, index):
        if not 0 <= index < self.shards:
            raise ValueError(f"Shard {index} is outside [0, {self.shards})")
        start = index * self.shard_size
        return start, min(start + self.shard_size, self.roots)

    def contains(self, index):
        return (
            self.db.execute(
                "SELECT 1 FROM shard WHERE ordinal = ?", (index,)
            ).fetchone()
            is not None
        )

    def register(self, index, rows, dropped):
        start, stop = self.bounds(index)
        if rows < 0 or rows + len(dropped) != stop - start:
            raise ValueError(f"Shard {index} does not reconcile with its root range")
        with self._transaction():
            try:
                self.db.execute(
                    "INSERT INTO shard VALUES (?, ?, ?)",
                    (index, rows, json.dumps(dropped, separators=(",", ":"))),
                )
            except sqlite3.IntegrityError as error:
                raise FileExistsError(
                    f"Shard is already registered: {index:06d}"
                ) from error
            self.db.execute(
                "UPDATE release SET done = done + 1, rows = rows + ?, drops = drops + ?",
                (rows, len(dropped)),
            )

    def counts(self):
        done, rows, drops = self.db.execute(
            "SELECT done, rows, drops FROM release WHERE id = 1"
        ).fetchone()
        return {
            "shards": done,
            "roots": rows,
            "dropped": drops,
            "pending": self.roots - rows - drops,
        }

    def finalize(self):
        counts = self.counts()
        if counts["shards"] != self.shards or counts["pending"]:
            return counts
        if not counts["roots"]:
            raise ValueError(
                "Every selected root was dropped; no release was published"
            )
        if self.release.is_file():
            return counts
        payload = {
            **self.metadata,
            "dataset_id": self.dataset_id,
            "curation": self.layout,
        }
        temporary = self.release.with_name(f".release-{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True)[:-1])
                for name, entries in (
                    ("shards", self._shards()),
                    ("dropped", self._drops()),
                ):
                    handle.write(f',\n"{name}":[\n')
                    for ordinal, entry in enumerate(entries):
                        if ordinal:
                            handle.write(",\n")
                        json.dump(entry, handle)
                    handle.write("\n]")
                handle.write("\n}\n")
            temporary.replace(self.release)
        finally:
            temporary.unlink(missing_ok=True)
        return counts

    def _shards(self):
        for index, rows in self.db.execute(
            "SELECT ordinal, rows FROM shard ORDER BY ordinal"
        ):
            path = f"shard-{index:06d}"
            if not (self.directory / path).is_dir():
                raise FileNotFoundError(
                    f"Registered curation shard is missing: {self.directory / path}"
                )
            yield {"path": path, "rows": rows}

    def _drops(self):
        for index, dropped in self.db.execute(
            "SELECT ordinal, dropped FROM shard WHERE dropped != '[]' ORDER BY ordinal"
        ):
            for root_id in json.loads(dropped):
                yield {"root_id": root_id, "shard": index}
