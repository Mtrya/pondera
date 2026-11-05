"""Ragged Parquet trace storage shared by HF and local training readers."""

import json
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pondera.chess.codec import BOARD_TOKENS_MAX
from pondera.chess.vocabulary import CHESS_TOKENIZER_VERSION
from pondera.data.trace import ROLE_COUNT, TRACE_GRAMMAR_VERSION

DATA_FORMAT_VERSION = 8
TRACE_ARRAYS = ("board_tokens", "trace_tokens", "trace_len", "trace_roles")
STATE_ARRAYS = (
    "state_without_board",
    "state_with_board",
    "state_best",
    "state_basis",
    "state_value",
)
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
    ]
)
TRACE_SCHEMA = pa.schema(
    [
        *HF_SCHEMA,
        ("root_id", pa.string()),
        ("split", pa.string()),
        ("effective_width", pa.int16()),
        ("rounds", pa.int16()),
        ("thinking_tokens", pa.int32()),
        ("trace_len", pa.int32()),
        ("final_value_bin", pa.int16()),
        ("claim_draw", pa.bool_()),
        ("position_key", pa.string()),
        ("source", pa.string()),
        ("board_tokens", pa.list_(pa.int16())),
        ("trace_tokens", pa.list_(pa.int16())),
        ("trace_roles", pa.list_(pa.int8())),
        *((name, pa.list_(pa.int32())) for name in STATE_ARRAYS),
    ]
)
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


def write_trace_rows(directory, rows):
    """Store unpadded rows with the HF scalars and authoritative machine arrays."""
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
    """Diagnostic checks of ragged array types, lengths and role ranges."""
    if not arrays:
        raise ValueError("A Trace dataset needs token arrays")
    row_count = len(next(iter(arrays.values())))
    if rows is not None and row_count != rows:
        raise ValueError("Trace arrays do not match the declared row count")
    for name, array in arrays.items():
        if len(array) != row_count:
            raise ValueError(f"Trace array {name} has a different row count")
        expected = (
            np.int16
            if name in {"board_tokens", "trace_tokens"}
            else np.int8
            if name == "trace_roles"
            else np.int32
        )
        if array.dtype != expected:
            raise ValueError(f"Trace array {name} has an unexpected dtype")
    boards = arrays.get("board_tokens")
    if boards is not None and np.any(boards.lengths > BOARD_TOKENS_MAX):
        raise ValueError("Board token array exceeds its declared width")
    trace = arrays.get("trace_tokens")
    if trace is not None and np.any(trace.lengths < 1):
        raise ValueError("Trace rows cannot be empty")
    lengths = arrays.get("trace_len")
    if (
        lengths is not None
        and trace is not None
        and not np.array_equal(lengths, trace.lengths)
    ):
        raise ValueError("Trace lengths do not match their examples")
    roles = arrays.get("trace_roles")
    if roles is not None:
        if trace is not None and not np.array_equal(roles.lengths, trace.lengths):
            raise ValueError("Trace roles do not match their examples")
        if np.any((roles.values < 0) | (roles.values >= ROLE_COUNT)):
            raise ValueError("Trace roles do not match their declared format")
    best = arrays.get("state_best")
    for name in STATE_ARRAYS:
        array = arrays.get(name)
        if (
            array is not None
            and best is not None
            and not np.array_equal(array.lengths, best.lengths)
        ):
            raise ValueError(f"Truncation-state array {name} has inconsistent lengths")


def trace_data_identity(directory):
    """The declared generation identity and shards of training input."""
    manifest = load_manifest(Path(directory) / "release.json")
    return {
        name: manifest.get(name)
        for name in (
            "format_version",
            "grammar_version",
            "tokenizer_version",
            "metadata",
            "shards",
        )
    }


class TracePositions:
    """Read machine columns directly from the same Parquet files used by HF."""

    CACHE_GROUPS = 8

    def __init__(self, directory, split):
        if split not in {"train", "val"}:
            raise ValueError(f"Unknown trace split: {split}")
        directory = Path(directory)
        self.manifest = load_manifest(directory / "release.json")
        for field, expected in (
            ("format_version", DATA_FORMAT_VERSION),
            ("tokenizer_version", CHESS_TOKENIZER_VERSION),
            ("grammar_version", TRACE_GRAMMAR_VERSION),
        ):
            if self.manifest.get(field) != expected:
                raise ValueError(
                    f"Unsupported dataset {field}: {self.manifest.get(field)}"
                )
        self.index = []
        self.shards = []
        self.cache = OrderedDict()
        for shard in self.manifest["shards"]:
            path = directory / shard["path"] / "data.parquet"
            file = pq.ParquetFile(path)
            if file.metadata.num_rows != shard["rows"]:
                raise ValueError(f"Shard row count differs from its release: {path}")
            starts, count = [], 0
            for group in range(file.num_row_groups):
                starts.append(count)
                count += file.metadata.row_group(group).num_rows
            splits = file.read(columns=["split"], use_threads=False)[
                "split"
            ].to_pylist()
            shard_index = len(self.shards)
            self.shards.append((path, starts))
            for row, label in enumerate(splits):
                if label not in {"train", "val"}:
                    raise ValueError(f"Unknown shard split: {label}")
                if label == split:
                    group = bisect_right(starts, row) - 1
                    self.index.append((shard_index, group, row - starts[group]))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        shard, group, row = self.index[index]
        key = (shard, group)
        if key not in self.cache:
            file = pq.ParquetFile(self.shards[shard][0])
            table = file.read_row_group(
                group,
                columns=["board_tokens", "trace_tokens", "trace_len", "trace_roles"],
                use_threads=False,
            )
            self.cache[key] = _arrays_from_table(table, TRACE_ARRAYS)
            if len(self.cache) > self.CACHE_GROUPS:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        arrays = self.cache[key]
        return (
            arrays["board_tokens"][row].copy(),
            arrays["trace_tokens"][row].copy(),
            int(arrays["trace_len"][row]),
            arrays["trace_roles"][row].copy(),
        )
