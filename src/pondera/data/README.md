# Offline data and trace pipeline

This layer prepares teacher traces and reads them for training. It does not import training, inference, or evaluation code. `python -m pondera.data` exposes `prepare`, `curate`, and `encode`, all controlled by `configs/data.toml`. Paths are command arguments.

| File | Responsibility |
|---|---|
| `__main__.py` | Three commands: provision/preprocess configured sources, curate prepared roots, and fit a vocabulary plus encode a dataset. |
| `config.py` | Shared TOML parsing, typed settings, and the sampling recipe contract; no runtime paths. |
| `roots/` | Source acquisition and root preparation: PGN replay with game history, Lichess puzzle setup moves, shared deduplication/splits/sampling, and self-contained provenance. See [roots/README.md](roots/README.md). |
| `trace.py` | Trace grammar and visit state machine; sidecar parsing and validation; translation of one completed search into trace rows; independent token replay. |
| `pipeline.py` | Joint width/depth sampling with independent presentation draws; worker engines with per-root search resets; curation from a prepared root pool; atomic shard publication and recovery; release statistics and independent validation. |
| `dataset.py` | Central local/HF schemas, dictionary-aware dataset identity, and projected row-group loading of stored IDs with derived initial boards. |
| `text.py` | Convert the authoritative SAN/control/BOARD transcription into base atoms and project diagnostic roles through dictionary expansions. |
| `tokenize.py` | Fit one configured dictionary and stream local or HF Parquet into a separate training dataset, replacing trace IDs and lengths while preserving other fields; also apply an existing dictionary through the library API. |
| `merge_profile.py` | Fit min-cost merge dictionaries on declared full-row corpora and measure stored presentations, optional held-out groups, compression slices, and resource costs. |

Preparation ends at the root pool; curation reads it without rebuilding or rewriting it. `pipeline.py` does not persist a collection or a sidecar directory. A root's sidecar exists only in a temporary directory while its engine process runs. `dataset.py` is the read-side boundary and does not depend on the production pipeline.
