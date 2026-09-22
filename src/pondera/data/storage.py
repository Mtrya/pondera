"""Shared manifests and storage for position datasets."""

import json
import shutil
import sqlite3
from collections import OrderedDict, defaultdict
from hashlib import blake2b
from pathlib import Path
from typing import Dict, Optional

from datasets import Dataset, DatasetDict, Features, Value, load_dataset
from huggingface_hub import create_repo

FINAL_FEATURES = Features(
    {
        "fen": Value("string"),
        "next_move": Value("string"),
        "count": Value("int64"),
    }
)


def stable_hash_int(text: str) -> int:
    return int.from_bytes(blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def write_manifest(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_manifest(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_clean_output_dir(output_dir: Path, overwrite: bool):
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    elif (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not (output_dir / "manifest.json").exists()
    ):
        raise FileExistsError(
            f"Refusing to reuse non-empty output directory without a manifest: {output_dir}. "
            "Pass --overwrite to clear it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def assemble_dataset(
    train_csv: Path,
    validation_csv: Path,
    dataset_dir: Path,
) -> DatasetDict:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset = DatasetDict(
        {
            "train": load_csv_split(train_csv),
            "validation": load_csv_split(validation_csv),
        }
    )
    # Empty splits need an explicit shard so they can be loaded from disk.
    dataset.save_to_disk(
        str(dataset_dir),
        num_shards={name: None if len(split) else 1 for name, split in dataset.items()},
    )
    return dataset


def load_csv_split(csv_path: Path) -> Dataset:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        next(handle, None)
        has_rows = next(handle, None) is not None

    if not has_rows:
        return Dataset.from_dict(
            {
                "fen": [],
                "next_move": [],
                "count": [],
            },
            features=FINAL_FEATURES,
        )

    return load_dataset(
        "csv",
        data_files=str(csv_path),
        features=FINAL_FEATURES,
        split="train",
    )


def push_dataset(
    dataset: DatasetDict,
    repo_id: str,
    token: Optional[str],
    private: bool,
    max_shard_size: str = "500MB",
):
    create_repo(
        repo_id=repo_id,
        token=token,
        private=private,
        repo_type="dataset",
        exist_ok=True,
    )
    dataset.push_to_hub(
        repo_id=repo_id, token=token, private=private, max_shard_size=max_shard_size
    )


class CountBuckets:
    """Aggregate position or position/move counts in bounded SQLite connections."""

    def __init__(self, bucket_dir: Path, by_move=False, max_open_dbs=64):
        self.bucket_dir = bucket_dir
        self.bucket_dir.mkdir(parents=True, exist_ok=True)
        self.keys = ("fen", "next_move") if by_move else ("fen",)
        self.max_open_dbs = max_open_dbs
        self.connections = OrderedDict()

    def _connection(self, bucket_id):
        if bucket_id in self.connections:
            connection = self.connections.pop(bucket_id)
        else:
            if len(self.connections) >= self.max_open_dbs:
                _, oldest = self.connections.popitem(last=False)
                oldest.commit()
                oldest.close()
            connection = sqlite3.connect(
                self.bucket_dir / f"bucket-{bucket_id:05d}.sqlite"
            )
            for setting in (
                "journal_mode=OFF",
                "synchronous=OFF",
                "temp_store=MEMORY",
                "locking_mode=EXCLUSIVE",
            ):
                connection.execute(f"PRAGMA {setting}")
            columns = ", ".join(f"{key} TEXT NOT NULL" for key in self.keys)
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS counts ({columns}, count INTEGER NOT NULL, PRIMARY KEY ({', '.join(self.keys)}))"
            )
        self.connections[bucket_id] = connection
        return connection

    def write_rows(self, rows):
        grouped = defaultdict(list)
        for bucket_id, *values in rows:
            grouped[bucket_id].append(values)
        keys = ", ".join(self.keys)
        placeholders = ", ".join("?" for _ in range(len(self.keys) + 1))
        statement = f"INSERT INTO counts ({keys}, count) VALUES ({placeholders}) ON CONFLICT ({keys}) DO UPDATE SET count = count + excluded.count"
        for bucket_id, values in grouped.items():
            connection = self._connection(bucket_id)
            connection.executemany(statement, values)
            connection.commit()
        return sum(map(len, grouped.values()))

    def close(self):
        for connection in self.connections.values():
            connection.commit()
            connection.close()
        self.connections.clear()
