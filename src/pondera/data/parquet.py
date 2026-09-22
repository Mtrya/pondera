"""Read and systematically sample local or Hub-hosted position shards."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files


def download_split(
    repo: str, split: str, cache_dir: Path, revision: str = "main"
) -> list[Path]:
    files = sorted(
        name
        for name in list_repo_files(repo, repo_type="dataset", revision=revision)
        if name.startswith(f"data/{split}-") and name.endswith(".parquet")
    )
    if not files:
        raise ValueError(f"No parquet shards for {repo}:{split}")

    def download(name):
        return Path(
            hf_hub_download(
                repo, name, repo_type="dataset", revision=revision, cache_dir=cache_dir
            )
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(download, files))


def sampled_batches(paths, columns, rows=None, batch_size=100_000):
    """Sample evenly over concatenated shards, preserving their supplied order."""
    counts = [pq.ParquetFile(path).metadata.num_rows for path in paths]
    total = sum(counts)
    if rows is None:
        rows = total
    if not 0 < rows <= total:
        raise ValueError(f"Requested {rows} rows from {total} available")
    selected = np.arange(rows, dtype=np.int64) * total // rows
    offset = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=batch_size, columns=columns
        ):
            lo, hi = np.searchsorted(selected, [offset, offset + batch.num_rows])
            if hi > lo:
                yield batch.take(pa.array(selected[lo:hi] - offset))
            offset += batch.num_rows
