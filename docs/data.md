# Data synthesis

This page describes the current offline data pipeline. It is an overview, not
a frozen format: the trace grammar, configuration keys, and array layout are
still changing, and `pondera.data.sidecar` plus the grammar code are the exact
schema authority. A release records the SHA-256 of its configuration and of
this document, so it remains interpretable against the version that produced
it.

Training never runs the engine. Offline, one root position becomes a causal
trace:

    root -> instrumented Stockfish search -> sidecar -> trace tokens
         -> sharded release -> validation

## Roots

`pondera.data.root_corpus` builds a root corpus from a pinned Lichess Elite
archive. Every game is replayed with python-chess; a position is keyed by its
first four FEN fields and keeps its last in-game occurrence, which carries the
highest repetition count and halfmove clock. Terminal positions and positions
seen five or more times are excluded. Splits come from a hash of the position
key. The `sample` subcommand draws a stratified pool (split × draw-claimable)
by hash rank, so a prefix of the pool is a uniform subsample.

A root record carries `root_id` (SHA-256 of `fen|repetition`), `split`, `fen`,
`repetition`, `turn`, `position_key`, and either a fixed teacher width/depth or
a per-root recipe sample.

## Teacher collection

The teacher is an external instrumented Stockfish 17.1 build. Its source and
binary are not part of this repository, and the repository contains no engine
source code. One fresh engine process runs per root: `Threads`, `Hash`, and
`MultiPV` are set, `TraceLog` is enabled, then `position fen` and
`go depth D`. No game history, time limit, or searchmoves are supplied.

The instrumented binary writes a JSONL sidecar. Schema 3 is used at width one;
schema 4 records the per-PV-line attempts at width greater than one. Events are
`session`, `position`, `search_start`, `round_start`, `attempt`, `candidate`,
`frontier`, `round_end`, and `search_end`. Values are integers: internal engine
units, with `cp` and WDL supplied for ordinary scores and mate constants in the
session event. The record is deterministic for a fixed binary and options.

`configs/trace.toml` is the single authority for collection and synthesis
settings. A `[recipe]` table may sample width, depth, and presentation per root
from `random.Random(f"{seed}:{root_id}:<table>")`; without it, every root uses
the fixed `[search]` settings and the release writes all four presentations.

## Trace synthesis

`pondera.data.sidecar` validates the record; `pondera.data.trace_synthesis`
translates it into grammar-v2 tokens. One root trace is built from the complete
search and then truncated for presentation.

A trace is a causal program over the root. `<depth>` opens a round at the root.
A move token enters the child reached by that legal move; `BACK` returns to the
parent; `RESET` returns to the root from any depth. Value and mate tokens
evaluate the current node from the root player's perspective. `<refuted>`
states that a just-entered candidate did not beat the leading exact score of
its parent in the current visit. `BOARD` requests the full board encoding of
the current node; the root never requests one because its board is the prefix.

A node visit runs from the entering move to the `BACK` or `RESET` that leaves
it, and records exactly one outcome: an exact evaluation, a refutation, or a
pass. Writing an evaluation or refutation ends the visit. A round's root
evaluation is its last operation, after which only a new `<depth>` or `FINAL`
is legal. Checkmate, stalemate, path repetition, and the 50-move rule stop the
search below the root; the 50-move rule is a claimable draw at the root.

`FINAL` is the decision point and implicitly returns to the root. It is
followed by a root move, a value token, an optional mate-distance token, the
best line, and `<end>`. When draw claims are enabled and the root is claimable,
the decision is `FINAL <claim_draw> <value> [<mate>] <end>` instead. The suffix
is a prediction region: a truncated trace carries the final answer of its
complete search.

The four presentations are `zero` (suffix only), `full` (complete thought),
`random` (one cut at an interior operation boundary), and `last_root` (the
boundary just before the final root evaluation). All presentations of a root
share the same label suffix.

## Release and validation

A release is a directory of shards plus a `release.json` manifest. Each shard
contains `board_tokens`, `trace_tokens`, `trace_len`, `trace_roles`, the
truncation-state arrays, `state_count`, `roots.jsonl`, and `champions.jsonl`,
with per-file hashes in the shard manifest. Shards are published atomically
and can be recovered or resumed by root.

Sequence datasets use manifest `format_version = 5` and `grammar_version = 2`.
Roles mark thought moves, evaluations, controls, refutations, the FINAL
decision, the label evaluation, and the best line. `pondera.data.positions`
reads these arrays for training; shard arrays are memory-mapped lazily so a
large release does not exhaust file descriptors.

`pondera.data.sequence_validation` checks release identity, manifest and array
shapes, shard checksums, stored truncation states, and every trace replay
against python-chess. Validation is deliberately independent of synthesis: it
reconstructs each program from the stored tokens and the root FEN.

## Scope and reproducibility

This repository ships the Python side of the pipeline: the configuration
schema, sidecar validator, trace synthesizer, release builder, release
validator, and training dataset adapters. It deliberately does not ship
Stockfish source, an instrumentation patch, a treatment binary, a control
binary, sidecars, or releases. Data synthesis is therefore not reproducible
from a clean checkout, by design; those artifacts are external.

Reproducing a release requires the external treatment and control builds, the
sidecars they produce, the pinned configuration, and the root pool. The
collector records the hashes of the engine binary, instrumentation patch,
configuration, and this document. The detailed instrumentation procedure is an
agent skill at `.agents/skills/stockfish-trace/SKILL.md`: prose only, no engine
source. It tells an agent how to build the external instrumentation and hand
the resulting sidecars back to this repository.

The Python entry points exist for experiments that already have those
artifacts; the repository does not provide a one-command reproduction path.

## Current limits

The mechanism is not final. Trace grammar, presentation options, and release
fields may change together; avoid building external consumers against the
current arrays. The instrumented teacher removes only wall-clock `time` and
`nps` when comparing builds, and the engine binary must be rebuilt from source
on the target platform rather than copied across incompatible systems.
