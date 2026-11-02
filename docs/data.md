# Data synthesis

This page describes the current offline data pipeline. The grammar and array layout are still changing; `pondera.data.trace` and `pondera.data.dataset` are the exact schema authorities.

Training never runs the engine. Offline production has one entry point, `pondera.data.pipeline`, and one configuration file:

    game archive -> stratified root pool -> instrumented Stockfish
                 -> in-memory sidecar checks, synthesis, and replay
                 -> sharded release -> independent release validation

## Roots

`pondera.data.roots` replays a pinned Lichess Elite archive with python-chess and builds the requested root pool directly. It does not write a full root corpus. A position is keyed by its first four FEN fields and keeps its last in-game occurrence, carrying the highest repetition count and halfmove clock. Across games, higher repetition wins and ties keep the first record. Terminal positions and positions seen five or more times are excluded. Splits come from a deterministic draw on the position key.

The pool is stratified by split and whether the root allows a draw claim. Ranked selection is reproducible and grows by extension; an independent draw determines output order. A root record carries `root_id` (`fen|repetition`), `split`, `fen`, `repetition`, `turn`, `position_key`, source provenance, and its sampled teacher settings and presentation.

## Configuration

`configs/trace.toml` is the single authority for production settings. `[paths]` locates the archive, cached root pool, external engine, and output. `[roots]` sets train/validation sizes and the draw-claimable share. `[search]` sets the engine hash-table size and independent width/depth distributions. `[synthesis]` sets BOARD probabilities, draw claims, presentation probabilities, and the random-cut interval. `[release]` sets shard size and worker count. The top-level `seed` determines root selection and all per-root samples.

Width, depth, and presentation accept either one value or a probability table. Each root samples them independently using `random.Random(f"{seed}:{root_id}:<setting>")`. Exactly one of `zero`, `full`, or `random` is materialized for each root. Assignments are recomputed from the active recipe and stable root ids: an identical recipe reproduces them on resume, while a changed recipe replaces them.

## Teacher and trace synthesis

The teacher is an external instrumented Stockfish 17.1 build. Neither its source nor its binary is part of this repository. One fresh engine process runs per root: `Threads`, `Hash`, and `MultiPV` are set, `TraceLog` is enabled, then `position fen` and `go depth D` are sent. No game history, time limit, or searchmoves are supplied.

The engine writes a JSONL sidecar in a temporary directory. Schema 3 is used at configured width one; schema 4 records per-PV-line attempts at greater widths. Events are `session`, `position`, `search_start`, `round_start`, `attempt`, `candidate`, `frontier`, `round_end`, and `search_end`. Scores use internal integer units, with `cp` and WDL for ordinary scores and mate constants in the session event.

For each root, `pondera.data.trace` validates the sidecar, translates the completed search into grammar-v2 tokens, and independently replays the selected row before publication. The sidecar and unsharded trace are not retained as pipeline artifacts.

A trace is a causal program over the root. `<depth>` opens a round. A move enters its child; `BACK` returns to the parent; `RESET` returns to the root. Values and mate tokens evaluate the current node from the root player's perspective. `<refuted>` says that a just-entered candidate did not beat the parent's current leading exact score. `BOARD` represents the current node; the root board is already the input prefix.

A node visit records an exact evaluation, a refutation, or a pass. An evaluation or refutation completes the visit. A round's root evaluation is its last operation, after which only a new `<depth>` or `FINAL` is legal. Checkmate, stalemate, path repetition, and the 50-move rule stop the search below the root; the 50-move rule is a claimable draw at the root.

`FINAL` implicitly returns to the root. It is followed by the chosen move, a value token, an optional mate-distance token, the best line, and `<end>`. An enabled root draw claim uses `FINAL <claim_draw> <value> [<mate>] <end>`. A truncated trace still carries the final answer of the complete search.

The presentations are `zero` (suffix only), `full` (complete thought), and `random` (one cut at an interior operation boundary).

## Release and validation

A release contains shards and `release.json`. Each shard stores `board_tokens`, `trace_tokens`, `trace_len`, `trace_roles`, the truncation-state arrays, `state_count`, `roots.jsonl`, and `champions.jsonl`. Shards are published atomically and can be recovered or resumed by root.

Trace datasets use manifest `format_version = 6` and `grammar_version = 2`. `pondera.data.dataset` reads these arrays for training and lazily memory-maps shard storage. `pondera.data.pipeline.validate_release` checks release identity, manifests, array shapes, stored states, and every trace replay against python-chess. Validation reconstructs programs from stored tokens rather than trusting synthesis.

## Scope and reproducibility

This repository ships root-pool construction, the Python sidecar consumer and synthesizer, release construction and validation, and training adapters. It deliberately ships no Stockfish source, instrumentation patch, engine binaries, sidecars, or releases. Data synthesis is therefore not reproducible from a clean checkout without external artifacts, by design.

The external instrumentation procedure is an agent skill at `.agents/skills/stockfish-trace/SKILL.md`: prose only, no engine source. It explains how to build and verify the teacher and connect it to the single pipeline entry point. Release identity records the engine name and effective generation settings.

Trace grammar, presentation options, and release fields may still change together; external consumers should not treat the current arrays as stable.
