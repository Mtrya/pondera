# Offline data and trace pipeline

This layer prepares teacher traces and reads them for training. It does not import training, inference, or evaluation code. `python -m pondera.data` exposes `prepare`, `curate`, and `encode`, all controlled by `configs/data.toml`. Paths are command arguments.

| File | Responsibility |
|---|---|
| `__main__.py` | Three commands: prepare configured sources, curate prepared roots, and fit a vocabulary plus encode directly to training memory maps. |
| `config.py` | Shared TOML parsing, typed settings, and the sampling recipe contract; no runtime paths. |
| `roots/` | Source acquisition and root preparation: PGN replay with game history, Lichess puzzle setup moves, shared deduplication/splits/sampling, and self-contained provenance. See [roots/README.md](roots/README.md). |
| `trace.py` | Trace grammar and visit state machine; sidecar parsing and validation; translation of one completed search into trace rows; independent token replay. |
| `pipeline.py` | Joint width/depth sampling with independent presentation draws; worker engines with per-root search resets; curation from a prepared root pool; per-root checkpoints, atomic shard publication, and independent validation. |
| `progress.py` | Per-shard curation progress, range reconciliation, and publication of the ordered release manifest. |
| `dataset.py` | Shared HF schema, local trace atom storage, and ordered discovery of completed release shards. |
| `trace_encoding.py` | Apply the same merge boundaries during fitting and encoding. |
| `tokenize.py` | Fit a dictionary and encode stored trace atoms into memory maps with reusable shard streams and separate ordered assembly. Save the completed fit beside the output for encoding retries. `encode_dataset` applies an existing dictionary. |
| `merge_profile.py` | Fit merge dictionaries on the selected corpus in release order and measure compression across the stored presentations, optional held-out groups, and resource costs. |
| `packing.py` | Read memory-mapped documents, select a seeded nested prefix under the configurable length cap, and pack whole documents from a consumed-document cursor into the training batch contract. |

Curation reads active ranges from the immutable root pool. Jobs can target disjoint half-open shard intervals in the same release directory. Each completed root stays in `.roots/` until its shard is registered. A timeout is recorded as a drop and any other collection failure gets up to three attempts across runs. Each shard is published atomically with its row count and dropped IDs, then registered in the shared `.curation.sqlite` ledger. A restart can register a published shard by reconciling it against the active root range. The completed `release.json` lists every shard in root-pool order and all recorded drops. The per-root synthesis diagnostics stay with the shards in `champions.jsonl`.

Fitting and encoding read the completed release. The encoder stages complete streams in a directory named `.<output>.encoded-shards` beside the output. Rerunning the same command reuses the fit and the completed shards for the same source and dictionary. Final assembly reads those shards in order and atomically publishes the packed directory before deleting the staged streams.
