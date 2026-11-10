"""Ragged Parquet trace storage shared by HF and local dataset pipelines."""

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA_FORMAT_VERSION = 11
TRACE_ARRAYS = ("trace_token_ids", "trace_len")
STATE_ARRAYS = ("state_best", "state_value")
DATA_ARRAYS = (*TRACE_ARRAYS, *STATE_ARRAYS, "state_count")
HF_SCHEMA = pa.schema(
    [
        ("fen", pa.string()),
        ("presentation", pa.string()),
        ("repetition", pa.int16()),
        ("turn", pa.string()),
        ("teacher_depth", pa.int16()),
        ("teacher_width", pa.int16()),
        ("trace_text", pa.string()),
        ("final_move", pa.string()),
        ("final_value", pa.float32()),
        ("mate_distance", pa.int16()),
        ("source", pa.string()),
    ]
)
LOCAL_ONLY_SCHEMA = pa.schema(
    [
        ("root_id", pa.string()),
        ("split", pa.string()),
        ("trace_token_ids", pa.list_(pa.int32())),
        ("trace_len", pa.int32()),
        ("state_best", pa.list_(pa.int32())),
        ("state_value", pa.list_(pa.int32())),
    ]
)
TRACE_SCHEMA = pa.schema([*HF_SCHEMA, *LOCAL_ONLY_SCHEMA])
HF_COLUMNS = tuple(field.name for field in HF_SCHEMA)
LOCAL_ONLY_COLUMNS = tuple(field.name for field in LOCAL_ONLY_SCHEMA)
ROOT_COLUMNS = [
    field.name
    for field in TRACE_SCHEMA
    if not pa.types.is_list(field.type) and field.name != "trace_text"
]


def write_manifest(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_manifest(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parquet_sources(directory):
    """List release shards or HF split files as (path, row count, split)."""
    directory = Path(directory)
    manifest = load_manifest(directory / "release.json") if directory.is_dir() else {}
    if manifest:
        result = []
        for shard in manifest["shards"]:
            path = directory / shard["path"]
            if path.suffix != ".parquet":
                path /= "data.parquet"
            result.append((path, shard["rows"], shard.get("split")))
        return result
    root = directory / "data" if (directory / "data").is_dir() else directory
    paths = [root] if root.is_file() else sorted(root.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet data in {directory}")
    splits = {"train": "train", "validation": "val", "val": "val"}
    return [
        (
            path,
            pq.ParquetFile(path).metadata.num_rows,
            splits.get(path.stem.split("-")[0]),
        )
        for path in paths
    ]


def write_trace_rows(directory, rows):
    """Store unpadded rows with shared fields and local machine arrays."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = [
        {
            **row,
            "source": json.dumps(row["source"], sort_keys=True)
            if row.get("source") is not None
            else None,
        }
        for row in rows
    ]
    table = pa.Table.from_pylist(records, schema=TRACE_SCHEMA)
    pq.write_table(
        table,
        directory / "data.parquet",
        compression="zstd",
        compression_level=3,
        row_group_size=64,
        use_dictionary=["presentation", "turn", "split", "final_move"],
    )


def read_root_rows(directory):
    rows = pq.read_table(
        Path(directory) / "data.parquet", columns=ROOT_COLUMNS, use_threads=False
    ).to_pylist()
    for row in rows:
        if row["source"] is not None:
            row["source"] = json.loads(row["source"])
    return rows


class RaggedArray:
    """Numpy views of an Arrow list column, addressed one row at a time."""

    def __init__(self, column):
        column = column.combine_chunks()
        self.values = column.values.to_numpy(zero_copy_only=False)
        self.offsets = column.offsets.to_numpy(zero_copy_only=False)
        self.dtype = self.values.dtype

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        return self.values[self.offsets[index] : self.offsets[index + 1]]

    @property
    def lengths(self):
        return np.diff(self.offsets)


def _arrays_from_table(table, names):
    result = {}
    for name in names:
        if name == "trace_len":
            result[name] = table[name].to_numpy()
            continue
        column = "state_best" if name == "state_count" else name
        array = RaggedArray(table[column])
        result[name] = array.lengths if name == "state_count" else array
    return result


def load_trace_arrays(directory, names=TRACE_ARRAYS):
    columns = list(
        dict.fromkeys("state_best" if name == "state_count" else name for name in names)
    )
    table = pq.read_table(
        Path(directory) / "data.parquet", columns=columns, use_threads=False
    )
    return _arrays_from_table(table, names)


def validate_trace_arrays(arrays, rows=None):
    """Diagnostic checks of ragged array types and lengths."""
    if not arrays:
        raise ValueError("A Trace dataset needs token arrays")
    row_count = len(next(iter(arrays.values())))
    if rows is not None and row_count != rows:
        raise ValueError("Trace arrays do not match the declared row count")
    for name, array in arrays.items():
        if len(array) != row_count:
            raise ValueError(f"Trace array {name} has a different row count")
        if array.dtype != np.int32:
            raise ValueError(f"Trace array {name} has an unexpected dtype")
    trace = arrays.get("trace_token_ids")
    if trace is not None and np.any(trace.lengths < 1):
        raise ValueError("Trace rows cannot be empty")
    lengths = arrays.get("trace_len")
    if (
        lengths is not None
        and trace is not None
        and not np.array_equal(lengths, trace.lengths)
    ):
        raise ValueError("Trace lengths do not match their examples")
    best = arrays.get("state_best")
    for name in STATE_ARRAYS:
        array = arrays.get(name)
        if (
            array is not None
            and best is not None
            and not np.array_equal(array.lengths, best.lengths)
        ):
            raise ValueError(f"Truncation-state array {name} has inconsistent lengths")
