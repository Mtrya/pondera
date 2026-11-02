# Offline data and trace pipeline

This layer prepares teacher traces and reads them for training. It does not import training, inference, or evaluation code. Production has one entry point, `python -m pondera.data.pipeline`, controlled by `configs/trace.toml`.

| File | Responsibility |
|---|---|
| `roots.py` | Replay PGN/zip/zst game archives, preserve repetition history, validate root records, and build the requested stratified train/validation pool directly. |
| `trace.py` | Trace grammar and visit state machine; sidecar parsing and validation; translation of one completed search into trace rows; independent token replay. |
| `pipeline.py` | TOML schema and per-root sampling; fresh-engine execution; the in-memory per-root pipeline; atomic shard publication and recovery; release statistics and independent validation. |
| `dataset.py` | On-disk array constants and checks, dataset identity, lazy sharded trace loading, and read-only training dataset adapters. |
| `merge_profile.py` | Measure min-cost merge dictionaries over full-presentation release rows and write compression profiles and dictionaries. |

`pipeline.py` composes roots and traces; it does not persist a collection or a sidecar directory. A root's sidecar exists only in a temporary directory while its engine process runs. `dataset.py` is the read-side boundary and does not depend on the production pipeline.
