# Root preparation

`prepare_pool` reads named `[sources.*]` tables from the shared data configuration and writes `roots.jsonl` plus `pool.manifest.json`. Each source declares its train/validation counts. A source with both counts zero is skipped without downloading. Local source files can be supplied as `prepare --input NAME=PATH`; download/cache locations remain runtime paths.

| File | Responsibility |
|---|---|
| `__init__.py` | Prepare configured source quotas and publish the combined pool. |
| `pool.py` | Root records, position keys, deterministic splits, sampling, and pool storage. |
| `pgn.py` | Download/check the configured archive, replay PGNs with repetition history, and build a claim-stratified sample. |
| `puzzles.py` | Download a pinned HF Parquet revision, apply the first puzzle move, and sample the resulting roots. |

Every puzzle contributes one candidate at the position after `Moves[0]`; the solution begins with `Moves[1]`. Complete original game history is unavailable, so puzzle roots use repetition one and declare that assumption in provenance. Their FEN clock is preserved, and no claimable-position quota is imposed. The full original puzzle row and HF revision are retained in the shared JSON `source` field.

Local puzzle inputs retain the complete original row but use a null revision: an arbitrary supplied Parquet file cannot attest to the configured Hub commit. Hub downloads retain the configured immutable revision.

Splits depend only on the position key. Duplicate positions within a puzzle source keep the first row. Sources are considered in configuration order; a later source excludes positions already selected by earlier sources before filling its quota. Selected records are globally ordered with the existing seeded pool-order draw. Source readers create provenance; the Parquet writer preserves it without adding a particular source's identity.
