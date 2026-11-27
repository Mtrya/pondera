# Root preparation

`prepare` reads the configured Elite PGN archive and puzzle prefix. Supply local inputs with `--input SOURCE=PATH`, or let the command retrieve and cache the configured sources. An omitted source table leaves that source out. `limit_games` bounds Elite replay for fixture runs. `[sources.puzzles].prefix_fraction` selects the floor of its fraction of the total raw Parquet row count, in sorted filename and stored row order. A locally supplied puzzle file retains the complete puzzle row and records a null Hub revision.

| File | Responsibility |
|---|---|
| `__init__.py` | Provision inputs, run both readers, and publish the combined pool. |
| `pgn.py` | Stream PGN games through Rust replay and retain eligible positions with their game and ply. |
| `puzzles.py` | Apply the setup move from each raw row in the ordered prefix and count retained roots by prefix. |
| `pool.py` | Deduplicate in a temporary SQLite index, assign splits and order, and publish and read the Parquet pool. |

Deduplication uses the first four canonical FEN fields. The highest repetition wins. On ties the first eligible record wins, with Elite before puzzles. Validation takes the `[roots].validation_count` lowest seeded position ranks. A separate seeded position rank orders the whole pool. Both ranks use the first eight bytes of SHA-256 over the seed, rank name, and position key. The position key breaks ties. `[roots].workers` parallelizes PGN replay while preserving archive order.

The completed output directory contains `roots.parquet` and `pool.manifest.json`. The manifest records the pool identity, counts, source provenance, and puzzle prefix census. Provenance shared by a source is stored once in the manifest. For a half-open range in published order, use `RootPool(path).iter_range(start, stop)`. The reader reconstructs the full FEN, `fen|repetition` ID, split, and provenance and loads at most one row group at a time. `len(pool)` gives the total count. Curation uses the range reader for prepared pools. Small external JSONL root lists are also supported.

Put `TMPDIR` and `SQLITE_TMPDIR` on job-local scratch for the temporary index and sort. The output stays on the destination filesystem and is published after both files are complete.
