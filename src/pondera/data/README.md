# Offline data and trace pipeline

This layer prepares teacher traces and reads them for training. It does not import training, inference, or evaluation code. Production has one entry point, `python -m pondera.data.pipeline`, controlled by `configs/trace.toml`.

| File | Responsibility |
|---|---|
| `roots.py` | Replay PGN/zip/zst game archives, preserve repetition history, validate root records, and build the requested stratified train/validation pool directly. |
| `trace.py` | Trace grammar and visit state machine; sidecar parsing and validation; translation of one completed search into trace rows; independent token replay. |
| `pipeline.py` | TOML schema and joint width/depth sampling with independent presentation draws; worker engines with per-root search resets; the in-memory per-root pipeline; atomic shard publication and recovery; release statistics and independent validation. |
| `dataset.py` | Central local/HF schemas, dictionary-aware dataset identity, and projected row-group loading of stored IDs with derived initial boards. |
| `text.py` | Convert the authoritative SAN/control/BOARD transcription into base atoms and project diagnostic roles through dictionary expansions. |
| `tokenize.py` | Stream local or HF Parquet into a separate training dataset, creating or replacing trace IDs and lengths under the selected dictionary while preserving other fields. |
| `merge_profile.py` | Fit min-cost merge dictionaries on declared full-row corpora and measure stored presentations, optional held-out groups, compression slices, and resource costs. |

`pipeline.py` composes roots and traces; it does not persist a collection or a sidecar directory. A root's sidecar exists only in a temporary directory while its engine process runs. `dataset.py` is the read-side boundary and does not depend on the production pipeline.
