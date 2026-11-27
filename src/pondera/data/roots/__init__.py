"""Prepare a combined root pool from PGN and ordered puzzle sources."""

from pathlib import Path

from pondera.data.roots.pool import index_pool, write_pool


def prepare_pool(config, output, *, inputs=None, cache=None):
    """Replay the configured sources, then split and order their unique positions."""
    from pondera.data.roots import pgn, puzzles

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(cache) if cache is not None else output.parent / "source"
    inputs = inputs or {}
    unknown = set(inputs) - {source.name for source in config.sources}
    if unknown:
        raise ValueError(f"Unknown input sources: {sorted(unknown)}")
    reports = {}
    with index_pool(config.seed) as index:
        for source in sorted(config.sources, key=lambda source: source.kind != "pgn"):
            local = inputs.get(source.name)
            if source.kind == "pgn":
                archive = (
                    pgn.verify_archive(local, source)
                    if local is not None
                    else pgn.ensure_archive(cache / source.name, source)
                )
                report = pgn.collect(archive, source, config.roots.workers, index)
            else:
                files = puzzles.parquet_files(source, cache / source.name, local=local)
                report = puzzles.collect(files, source, index, local=local is not None)
            reports[source.name] = report
        return write_pool(
            output,
            index,
            {
                "seed": config.seed,
                "validation_count": config.roots.validation_count,
                "source_order": list(reports),
                "sources": reports,
            },
        )
