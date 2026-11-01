# Offline data and trace pipeline

The offline data layer. It owns teacher records, trace grammar, release
construction, validation, and the training dataset adapters; it does not
import training, inference, or evaluation code.

| File | Responsibility |
|---|---|
| `sidecar.py` | Parse and validate one schema-3 or schema-4 instrumented Stockfish record. This module checks the record against `docs/teacher.md`; it does not translate it. |
| `trace_synthesis.py` | Translate a validated sidecar into grammar-v2 trace tokens: rounds, refutations, draw claims, truncation boundaries, and the four presentations. |
| `trace_profile.py` | Streaming statistics over synthesized roots: lengths, champion overturns, refutation density, claim counts, and value trajectories. |
| `trace.py` | The trace grammar itself: `TraceWriter`, `replay`, the shared visit state machine, roles, and the label suffix. |
| `trace_config.py` | The TOML configuration schema, value validation, and per-root width/depth/presentation sampling. |
| `trace_collect.py` | Select roots, drive one fresh instrumented engine process per root, write sidecars, and maintain a resumable collection manifest. |
| `sequence_release.py` | Build sharded, independently hashed releases from a collection; select presentations, publish or recover shards, and aggregate the statistics document. |
| `sequence_validation.py` | Verify release identity, manifest and array layout, shard checksums, stored truncation states, and every trace replay against python-chess. |
| `sequence_format.py` | The on-disk sequence format constants: format version, trace arrays, and state arrays. |
| `positions.py` | Training dataset adapters: depth targets, sequence traces, the lazy sharded loader, and the training-input identity check. |
| `root_corpus.py` | Build a root corpus from a pinned Lichess Elite archive, assign train/validation splits, and sample a stratified root pool. |
| `merge_profile.py` | Profile merge dictionaries over release rows: compression curves, top composites, and dictionary files. |
| `storage.py` | JSON manifest loading and atomic writing, file hashing, and content hashes over mappings. |
