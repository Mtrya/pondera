# Offline data and trace pipeline

This layer prepares teacher traces and reads them for training. It does not import training, inference, or evaluation code. `python -m pondera.data` exposes `prepare`, `curate`, and `encode`, all controlled by `configs/data.toml`. Paths are command arguments.

| File | Responsibility |
|---|---|
| `__main__.py` | Three commands: prepare configured sources, curate prepared roots, and fit a vocabulary plus encode directly to training memory maps. |
| `config.py` | Shared TOML parsing, typed settings, and the sampling recipe contract; no runtime paths. |
| `roots/` | Source acquisition and root preparation: PGN replay with game history, Lichess puzzle setup moves, shared deduplication/splits/sampling, and self-contained provenance. See [roots/README.md](roots/README.md). |
| `trace.py` | Trace grammar and visit state machine; sidecar parsing and validation; translation of one completed search into trace rows; independent token replay. |
| `pipeline.py` | Joint width/depth sampling with independent presentation draws; worker engines with per-root search resets; curation from a prepared root pool; atomic shard publication and recovery; release statistics and independent validation. |
| `dataset.py` | Shared text-only HF schema, local base-atom caches, and projected loading for release validation. |
| `text.py` | Convert authoritative text to base atoms, apply shared merge boundaries, and project diagnostic roles through dictionary expansions. |
| `tokenize.py` | Fit a dictionary and encode local or HF text directly into final memory maps with parallel shard workers and atomic publication. Save the completed fit beside the output for encoding retries; `encode_dataset` applies an existing dictionary. |
| `merge_profile.py` | Fit min-cost merge dictionaries on declared full-row corpora and measure stored presentations, optional held-out groups, compression slices, and resource costs. |
| `packing.py` | Read memory-mapped documents, select a seeded nested prefix under the configurable length cap, and pack whole documents from a consumed-document cursor into the training batch contract. |

Preparation ends at the root pool; curation reads it without rebuilding or rewriting it. `pipeline.py` persists no sidecars — a root's sidecar exists only in a temporary directory while its engine process runs — but it checkpoints each synthesized root under `.roots/` beside the release until the root's shard is registered, so an interrupted run recollects only missing roots; the cache is removed once the whole release completes. A root whose engine exceeds `[search] timeout_seconds` is dropped without retry and recorded under `dropped` in `release.json`; any other collection failure is isolated and retried (three attempts across runs) before it may fail the run. Engines optionally probe Syzygy tablebases (`[search] syzygy_path`/`syzygy_probe_limit`); the per-root timeout, the probe limit, and a content fingerprint of the table directory join the release search identity. `dataset.py` is the read-side boundary and does not depend on the production pipeline.
