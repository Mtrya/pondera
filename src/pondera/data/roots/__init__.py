"""Prepare a root pool from configured PGN and Lichess puzzle sources."""

from pathlib import Path

from pondera.data.roots.pool import _pool_priority, write_pool


def prepare_pool(config, output, *, inputs=None, cache=None):
    """Select each source's quota, excluding positions selected earlier."""
    from pondera.data.roots import pgn, puzzles

    output = Path(output)
    cache = Path(cache) if cache is not None else output / "source"
    inputs = inputs or {}
    unknown = set(inputs) - {source.name for source in config.sources}
    if unknown:
        raise ValueError(f"Unknown input sources: {sorted(unknown)}")
    records, reports, excluded = [], {}, set()
    active = [source for source in config.sources if source.train + source.val]
    for index, source in enumerate(active):
        local = inputs.get(source.name)
        if source.kind == "pgn":
            archive = (
                pgn.verify_archive(local, source)
                if local is not None
                else pgn.ensure_archive(cache / source.name, source)
            )
            selected, report = pgn.build_pool(
                archive,
                source.train,
                source.val,
                source.claimable_share,
                config.seed,
                limit_games=source.limit_games,
                source={"archive_url": source.url, "archive_sha256": source.sha256},
                exclude=excluded,
            )
        else:
            files = puzzles.parquet_files(source, cache / source.name, local=local)
            selected, report = puzzles.build_pool(
                files, source, config.seed, exclude=excluded
            )
        records.extend(selected)
        if index + 1 < len(active):
            excluded.update(record["position_key"] for record in selected)
        reports[source.name] = report
    if len(active) > 1:
        records.sort(
            key=lambda record: _pool_priority(
                config.seed, record["root_id"], "pool-order"
            )
        )
    manifest = {
        "kind": "root-pool",
        "format_version": 3,
        "seed": config.seed,
        "parameters": {
            "train": sum(source.train for source in config.sources),
            "val": sum(source.val for source in config.sources),
        },
        "counts": {
            split: sum(record["split"] == split for record in records)
            for split in ("train", "val")
        },
        "source_order": [source.name for source in active],
        "sources": reports,
    }
    write_pool(output, records, manifest)
    return manifest
