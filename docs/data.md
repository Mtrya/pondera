# Data synthesis

This page describes the current offline data pipeline. The grammar and array layout are still changing; `pondera.data.trace` and `pondera.data.dataset` are the exact schema authorities.

Training never runs the engine. Offline production has one entry point, `pondera.data.pipeline`, and one configuration file:

    game archive -> stratified root pool -> instrumented Stockfish
                 -> streamed sidecar decoding and synthesis
                 -> Parquet release -> training and merge profiling

## Roots

`pondera.data.roots` replays a pinned Lichess Elite archive with python-chess and builds the requested root pool directly. It does not write a full root corpus. A position is keyed by its first four FEN fields and keeps its last in-game occurrence, carrying the highest repetition count and halfmove clock. Across games, higher repetition wins and ties keep the first record. Terminal positions and positions seen five or more times are excluded. Splits come from a deterministic draw on the position key.

The pool is stratified by split and whether the root allows a draw claim. Ranked selection is reproducible and grows by extension; an independent draw determines output order. A root record carries `root_id` (`fen|repetition`), `split`, `fen`, `repetition`, `turn`, `position_key`, source provenance, and its sampled teacher settings and presentation.

## Source provenance

A root's `source` records the PGN member filename, a zero-based game index, and the number of played half-moves (`ply`) leading to the root. For example, `{"file": "lichess_elite_2025-11.pgn", "game": 12460, "ply": 54}` identifies the position after 54 half-moves in the 12,461st game. The filename ends in `.pgn`, the chess game notation format.

The public archive URL and expected SHA256 are tracked as `SOURCE_URL` and `SOURCE_SHA256` in [`pondera.data.roots`](../src/pondera/data/roots.py). `ensure_archive(directory)` downloads and verifies that archive without requiring an existing corpus or research files. Open the named PGN member, count every game from zero before applying any filtering, read the indexed game, and replay the specified number of half-moves from its starting position. This recovers the original game and the history behind the root's FEN and repetition count. The current archive is [Lichess Elite November 2025](https://database.nikonoel.fr/lichess_elite_2025-11.zip).

## Configuration

`configs/trace.toml` is the single authority for production settings. `[paths]` locates the archive, cached root pool, external engine, and output. `[roots]` sets train/validation sizes and the draw-claimable share. `[search]` sets the engine hash-table size and a joint width/depth distribution. `[synthesis]` sets BOARD probabilities, draw claims, presentation probabilities, and the random-cut interval. `[release]` sets shard size and worker count. The top-level `seed` determines root selection and all per-root samples.

`[search].joint` is a nonempty array of `{width, depth, probability}` tables. Width is a positive integer or `"full"`; depth is a positive integer. Pairs must be unique, probabilities must lie in `[0, 1]`, and their sum must be one. A fixed setting is a one-entry array with probability one. For example:

```toml
[search]
hash_mb = 256
joint = [
    { width = "full", depth = 8, probability = 0.5 },
    { width = 1, depth = 24, probability = 0.5 },
]
```

Each root draws one pair using `random.Random(f"{seed}:{root_id}:search")`. Entries are canonicalized before sampling, so reordering the tables preserves assignments and recipe identity. `"full"` resolves to the number of legal root moves before setting UCI `MultiPV`. The root's `teacher_width` records that concrete configured width; `effective_width` records its minimum with the legal move count. An explicit width can therefore exceed the effective width on a constrained root. A full-width root with one legal move configures width one.

Presentation accepts one value or a probability table and uses a separate `random.Random(f"{seed}:{root_id}:presentation")` draw. Exactly one of `zero`, `full`, or `random` is materialized for each root. Assignments are recomputed from the active recipe and stable root ids, independently of input order, worker count, and shard boundaries. Cached settings are replaced by the active recipe; resuming an existing release with a different recipe is rejected.

## Teacher and trace synthesis

The teacher is an external instrumented Stockfish 17.1 build. Neither its source nor its binary is part of this repository. Each worker owns one engine process with fixed `Threads = 1` and `Hash`. For every root it closes the previous `TraceLog`, sends `ucinewgame` to clear search state, sets `MultiPV`, opens a new `TraceLog`, waits for `readyok`, and sends `position fen` and `go depth D`. Reset qualification compares recorded events with fresh-process searches across different roots and widths. No game history, time limit, or searchmoves are supplied.

The engine writes a JSONL sidecar in a temporary directory. Schema 3 is used at configured width one; schema 4 records per-PV-line attempts at greater widths. Events are `session`, `position`, `search_start`, `round_start`, `attempt`, `candidate`, `frontier`, `round_end`, and `search_end`. Scores use internal integer units, with `cp` and WDL for ordinary scores and mate constants in the session event.

For each root, the pipeline decodes the qualified producer's completed sidecar directly from the temporary file and checks search completion and requested settings. `pondera.data.trace` translates known search operations into grammar-v2 tokens, roles, truncation states, and SAN text together. It materializes the assigned presentation while retaining statistics about the complete search. Exhaustive sidecar parsing and token replay are separate development and incident diagnostics.

A trace is a causal program over the root. `<depth>` opens a round. A move enters its child; `BACK` returns to the parent; `RESET` returns to the root. Values and mate tokens evaluate the current node from the root player's perspective. `<refuted>` says that a just-entered candidate did not beat the parent's current leading exact score. `BOARD` represents the current node; the root board is already the input prefix.

A node visit records an exact evaluation, a refutation, or a pass. An evaluation or refutation completes the visit. A round's root evaluation is its last operation, after which only a new `<depth>` or `FINAL` is legal. Checkmate, stalemate, path repetition, and the 50-move rule stop the search below the root; the 50-move rule is a claimable draw at the root.

`FINAL` implicitly returns to the root. It is followed by the chosen move, a value token, an optional mate-distance token, the best line, and `<end>`. An enabled root draw claim uses `FINAL <claim_draw> <value> [<mate>] <end>`. A truncated trace still carries the final answer of the complete search.

The presentations are `zero` (suffix only), `full` (complete thought), and `random` (one cut at an interior operation boundary).

## Release and validation

A release contains shards and `release.json`. Each shard stores `data.parquet`, `champions.jsonl`, and its manifest. The local Parquet schema has seventeen columns: eleven shared fields plus `root_id`, `split`, `trace_token_ids`, `trace_len`, `state_best`, and `state_value`. Base token IDs and the two best-state lists are ragged; padding is added only when forming a training batch. `trace_len` supports scalar length queries, and state counts come from list offsets. Initial board tokens are derived from FEN and repetition, while token roles and thought-operation boundaries are derived from token structure. Shards are published atomically, registered by index and row count, and can be recovered or resumed against the selected roots. A persistent worker pool processes at most two shards at a time while completed shards are written.

Trace datasets use manifest `format_version = 9` and `grammar_version = 2`. The release identity includes the joint recipe and presentation probabilities; rows store teacher depth and configured width, with effective width derived from legal moves when needed. The shared columns are `fen`, `presentation`, `repetition`, `turn`, `teacher_depth`, `teacher_width`, `trace_text`, `final_move`, `final_value`, `mate_distance`, and `source`. `trace_text` transcribes the selected trace in absolute SAN, including board observations, evaluations, control operations, and the final suffix. `final_value` is nullable float32 `(W-L)/1000` from the deepest teacher WDL; mates use a null value and a signed mate distance. This scalar is distinct from the quantized value represented by trace tokens. Depth, width, repetition, and mate distance use int16; the remaining shared fields are strings.

`tools/export_hf.py` reads an accepted local release in bounded Arrow batches and writes separate train and validation Parquet files containing exactly the eleven shared columns. It preserves stored trace text and scalar values, partitions on local `split`, and omits all six local-only columns. `pondera.data.dataset.TracePositions` reads base token IDs from the local release and caches up to eight row groups; it leaves text and best-state columns unread during training. `merge_profile.py` reads full-presentation base token IDs directly. Source provenance is stored as a self-contained JSON string with the public archive URL and digest, PGN member, zero-based game index, and ply.

For bounded qualification or failure diagnosis, `pondera.data.pipeline.validate_release` reconstructs sampled settings from the recipe and seed, checks effective width against legal moves, and verifies metadata, ragged array types and lengths, stored states, and every trace replay against python-chess. Normal curation uses completion and publication checks. Older manifest formats are rejected. Parallel diagnostics use spawned workers and require an importable script with a main guard; use `workers=1` from stdin.

## Scope and reproducibility

This repository ships root-pool construction, the Python sidecar consumer and synthesizer, release construction and validation, and training adapters. It deliberately ships no Stockfish source, instrumentation patch, engine binaries, sidecars, or releases. Data synthesis is therefore not reproducible from a clean checkout without external artifacts, by design.

The external instrumentation procedure is an agent skill at `.agents/skills/stockfish-trace/SKILL.md`: prose only, no engine source. It explains how to build and verify the teacher and connect it to the single pipeline entry point. Release identity records the engine name and effective generation settings.

Trace grammar, presentation options, and release fields may still change together; external consumers should not treat the current arrays as stable.
