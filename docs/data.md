# Data synthesis

This page describes the current offline data pipeline. The grammar and array layout are still changing; `pondera.data.trace` and `pondera.data.dataset` are the exact schema authorities.

Training never runs the engine. Data preparation uses `python -m pondera.data` with three commands sharing `configs/data.toml`:

    game archive -> stratified root pool -> instrumented Stockfish
                 -> streamed sidecar decoding and synthesis
                 -> Parquet release -> merge fitting and encoding -> training

```bash
python -m pondera.data prepare --config configs/data.toml --output ROOTS
python -m pondera.data curate --config configs/data.toml --input ROOTS --output TRACES --engine STOCKFISH
python -m pondera.data encode --config configs/data.toml --input TRACES --output TRAINING
```

`prepare` provisions the enabled sources into `ROOTS/source/` and builds `roots.jsonl`; `--cache CACHE` changes the local download location, and repeated `--input NAME=PATH` arguments supply existing inputs for named sources. It does not start an engine. `curate` reads a prepared directory or root JSONL file and produces traces without rewriting the pool. `encode` fits one merge dictionary at the configured total vocabulary size and encodes the complete input dataset in the same workflow. Local inputs, outputs, cache, and the engine executable are command arguments; source identities, recipe, and execution settings come from TOML.

## Roots

`pondera.data.roots.pgn` replays the configured PGN archive with python-chess and builds the requested root pool directly. It does not write a full root corpus. A position is keyed by its first four FEN fields and keeps its last in-game occurrence, carrying the highest repetition count and halfmove clock. Across games, higher repetition wins and ties keep the first record. Terminal positions and positions seen five or more times are excluded. Splits come from a deterministic draw on the position key.

PGN selection is stratified by split and whether the root allows a draw claim. Ranked selection is reproducible and grows by extension; an independent draw determines output order. A prepared root record carries `root_id` (`fen|repetition`), `split`, `fen`, `repetition`, `turn`, `position_key`, and source provenance. Curation derives teacher settings and presentation in memory, leaving the prepared records unchanged.

`pondera.data.roots.puzzles` reads the pinned `Lichess/chess-puzzles` Parquet source. The source FEN precedes the opponent move; applying the first UCI move produces the actual puzzle start. Only that position becomes a candidate, rather than every solution ply. Puzzle roots retain the resulting FEN, including its halfmove clock, and use repetition one because the row does not contain full game history. Provenance records that assumption. Puzzles are sampled by split without a claimable-position quota. Stockfish produces their traces and final labels through the same curation path as PGN roots.

Each source fills its own configured counts. Sources are processed in TOML order; later sources exclude position keys already selected earlier before filling their quotas. Within puzzle data, duplicate positions keep the first row. The shared split rule keeps identical positions in the same split across sources.

## Source provenance

A root's `source` records the PGN member filename, a zero-based game index, and the number of played half-moves (`ply`) leading to the root. For example, `{"file": "lichess_elite_2025-11.pgn", "game": 12460, "ply": 54}` identifies the position after 54 half-moves in the 12,461st game. The filename ends in `.pgn`, the chess game notation format.

The public archive URL and expected SHA256 come from `[sources.elite]` in [`configs/data.toml`](../configs/data.toml). Preparation checks downloaded or supplied archive bytes once and writes that identity into each selected root's provenance. Open the named PGN member, count every game from zero before applying any filtering, read the indexed game, and replay the specified number of half-moves from its starting position. This recovers the original game and the history behind the root's FEN and repetition count. The reference archive is [Lichess Elite November 2025](https://database.nikonoel.fr/lichess_elite_2025-11.zip).

Puzzle provenance contains the source kind, repository and immutable revision, the complete original puzzle row, `root_after_moves = 1`, and `repetition_is_assumed = true`. The stored FEN and setup move reproduce the root without local files; `PuzzleId` and `GameId` also identify the public puzzle/game. The local and HF Parquet writers preserve this JSON rather than inject the PGN archive identity.

## Configuration

`configs/data.toml` is the single authority for preparation, curation, and merge/encoding settings. Named `[sources.*]` tables declare source kind and train/validation counts; zero counts disable a source. PGNs specify `url`, `sha256`, `claimable_share`, and an optional game limit. Lichess puzzles specify `repo_id`, an immutable commit in `revision`, and Parquet batch size. The reference config keeps 500,000/2,000 PGN roots and zero puzzle roots; set the puzzle counts to include them. `[search]` sets engine hash-table size, the per-root timeout, a joint width/depth distribution, and optional Syzygy tablebase probing (`syzygy_path`, `syzygy_probe_limit`). `[synthesis]` sets BOARD probabilities, draw claims, presentation probabilities, and the random-cut interval. `[release]` sets shard size, worker count, and the shard range (`start_shard` inclusive, optional `end_shard` exclusive). `[merge]` sets the total vocabulary size, fitting corpus, candidate/selection settings, reporting size, and encoding batch size. The top-level `seed` determines root selection and all per-root samples. Runtime paths are not stored in the recipe.

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

A producer release contains shards and `release.json`. Each shard stores `data.parquet`, `champions.jsonl`, and its manifest. The local Parquet schema has seventeen columns: eleven shared fields plus `root_id`, `split`, `trace_token_ids`, `trace_len`, `state_best`, and `state_value`. Trace IDs and the two best-state lists are ragged; padding is added only when forming a training batch. `trace_len` counts the stored IDs, including the final suffix and excluding the initial board prefix. State counts come from list offsets, and their ordinal alignment follows complete thought operations in the expanded base sequence. Initial board tokens derive from FEN and repetition. Shards are published atomically, registered by index and row count, and can be recovered or resumed against the selected roots. A persistent worker pool refills root searches independently of shard completion, with at most twice the worker count in flight. Each worker checkpoints its result before reporting completion; shard publication loads records from those checkpoints.

Checkpoints remain under `.roots/` until their shard is registered. Roots exceeding the search timeout are recorded under `dropped` in `release.json` and are never retried. Other collection failures get up to three attempts across runs; a shard with unresolved failures stays unpublished. Engine startup failures abort the run without consuming a root attempt. An abort cancels queued searches and interrupts running engines while preserving completed checkpoints. A shard whose roots all time out contains an empty Parquet file, which encoding preserves.

Trace datasets use manifest `format_version = 11` and `grammar_version = 2`; stored trace IDs use list<int32>. The release identity includes the joint recipe and presentation probabilities; rows store teacher depth and configured width, with effective width derived from legal moves when needed. The shared columns are `fen`, `presentation`, `repetition`, `turn`, `teacher_depth`, `teacher_width`, `trace_text`, `final_move`, `final_value`, `mate_distance`, and `source`. `trace_text` is authoritative and transcribes the selected trace in absolute SAN, including board observations, evaluations, control operations, and the final suffix. `final_value` is nullable float32 `(W-L)/1000` from the deepest teacher WDL; mates use a null value and a signed mate distance. This scalar is distinct from the quantized value represented by trace tokens. Depth, width, repetition, and mate distance use int16; the remaining shared fields are strings.

`tools/export_hf.py` reads an accepted local release in bounded Arrow batches and writes Parquet files for the train/validation splits present, containing exactly the eleven shared columns. It preserves stored trace text, scalar values, and source JSON, partitions on local `split`, and omits all six local-only columns. `pondera.data.dataset.TracePositions` reads the selected trace encoding and caches up to eight row groups; it leaves text and best-state columns unread during training. Source provenance is a self-contained JSON string describing the configured PGN or puzzle origin.

For bounded qualification or failure diagnosis, `pondera.data.pipeline.validate_release` reconstructs sampled settings from the recipe and seed, checks effective width against legal moves, and verifies metadata, ragged array types and lengths, stored states, and every trace replay against python-chess. Normal curation uses completion and publication checks. Older manifest formats are rejected. Parallel diagnostics use spawned workers and require an importable script with a main guard; use `workers=1` from stdin.

## Merge vocabulary and training encoding

`python -m pondera.data encode` counts base-atom substrings on declared full-presentation rows, fits the configured `[merge].vocab_size`, measures compression, and writes the tokenized training artifact with its selected dictionary. `vocab_size` is the total vocabulary size including the `CHESS_VOCAB_SIZE` base atoms; the fitter may select fewer composites than the difference allows. A `vocab_size` equal to the base vocabulary gives base encoding without requiring full-presentation fitting rows. There is no automatic search for a winning size.

`[merge].fit_split` and `holdout_fraction` can select fitting and held-out groups; holdout is disabled by default. Optional `max_roots` declares deterministic fitting/measurement subsampling. These choices affect vocabulary fitting, while encoding preserves every input row and downstream split label. The output's `profile/curve.json` and `profile/profile.md` report the configured settings, actual fitting rows, atoms, selection sample, candidate limits, compression slices, time, and memory. `build_profile` remains a library API for calibration across several vocabulary sizes.

Encoding accepts current local release, format-9 base-ID producer data, or shared HF Parquet and publishes a separate training artifact after fitting and conversion complete. Both profiling and encoding read authoritative text; stale cached IDs do not affect the fitted dictionary. Encoding creates or replaces `trace_token_ids`, updates `trace_len`, and stores the selected flat expansions once in `release.json` under `merge_entries`. Shared values, row order, split membership, and existing teacher annotations pass through unchanged. HF-derived artifacts add the required ID/length fields and retain split membership in the manifest; they do not fabricate absent best-state annotations. Text parsing interprets SAN navigation and BOARD observations without running a teacher search or strict grammar replay. `tokenize_dataset` is available to library callers applying an existing dictionary. A training artifact receives a `dataset_id`, preserved during copying or re-encoding and included in resume identity, so independently imported datasets cannot share a resume identity merely because their shard sizes match.

Base token IDs and names are generated by `pondera.chess.vocabulary`, with tokenizer version 3 preserving their existing order. `merge_entries[i]` contains 2–8 base atom IDs and defines composite ID `CHESS_VOCAB_SIZE + i`; absent or empty entries indicate base encoding. Encoding minimizes the number of stored tokens. Initial board inputs remain base atoms. Training reads compressed IDs directly and masks padding; it does not derive diagnostic roles or replay the trace. Evaluation interprets dictionary expansions for per-role and final-token diagnostics. Equal next-token CE applies once per stored non-padding token. Mixed-role composites have a separate diagnostic category; a composite containing the final decision has the final-decision role. `final_token_top1` measures the token containing that decision, which can also contain surrounding atoms. Dataset identity and model checkpoints retain the selected expansions so resume and reload use the same encoding. Dictionary loading happens at setup; the training loop needs no vocabulary-file access. Calibration through `write_artifacts` stores each candidate's entries alongside its measurements in `curve.json`.

## Scope and reproducibility

This repository ships root-pool construction, the Python sidecar consumer and synthesizer, release construction and validation, and training adapters. It deliberately ships no Stockfish source, instrumentation patch, engine binaries, sidecars, or releases. Data synthesis is therefore not reproducible from a clean checkout without external artifacts, by design.

The external instrumentation procedure is an agent skill at `.agents/skills/stockfish-trace/SKILL.md`: prose only, no engine source. It explains how to build and verify the teacher and connect it to the single pipeline entry point. Release identity records the engine name and effective generation settings.

Trace grammar, presentation options, and release fields may still change together; external consumers should not treat the current arrays as stable.
