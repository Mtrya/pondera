---
name: stockfish-trace
description: Implement, build, and verify the external instrumented Stockfish 17.1 producer that emits Pondera's trace sidecars, then connect it to Pondera's single data pipeline. Use when building a teacher producer from a clean Pondera checkout or extending offline teacher production; never add Stockfish source, patches, or binaries to the Pondera repository.
---

# Stockfish trace producer

Pondera ships only the Python consumer under `src/pondera/data/`: root-pool construction, configuration, sidecar validation, trace synthesis and replay, release construction and validation, and dataset adapters. It does not ship Stockfish source, an instrumentation patch, engine binaries, sidecars, or releases.

When you start from a clean Pondera clone, build the producer outside the repository from upstream Stockfish 17.1 and this specification. `src/pondera/data/trace.py` is the consumer contract: if `parse_sidecar`, `synthesize_root`, or `replay` rejects your output, fix the producer. Do not copy Stockfish code into this skill or into the repository.

## 1. External workspace and control binary

Create a working directory outside the Pondera checkout (or under gitignored `.local/`).

1. Fetch official Stockfish `sf_17.1` source and ensure the default NNUE networks required by that tag are present in `src/`.
2. Copy the source tree into two sibling directories: an unmodified control tree and a treatment tree.
3. Build the control binary from the unmodified tree (`make -C src build ARCH=...` or `profile-build` with the same target flags you will use for the treatment). Keep the control binary for the consistency gates below.

## 2. Treatment instrumentation architecture

Instrument the treatment tree with a sidecar logger that is strictly observational: it must never alter search control flow, evaluation, move ordering, TT probes/writes, history tables, node counts, or standard UCI stdout.

### 2.1 UCI option and lifecycle

Register a string UCI option `TraceLog` (default `""`) in the engine option table.

- When `TraceLog` is set to `""`, close any open sidecar and reset trace state.
- When `TraceLog` is set to a non-empty path:
  - Close any previous sidecar first.
  - Check `Threads`: tracing is defined only for `Threads == 1`. If `Threads != 1`, leave tracing disabled and do not open or create the file.
  - Open the path in binary write/truncate mode. If opening fails, leave tracing disabled and write nothing to stdout/stderr.
  - Snapshot the configured `MultiPV` (`traceMultiPV = int(options["MultiPV"])`), reset `seq = 0`, and immediately write the `session` event.
- Also guard the start of search: if `Trace::active()` is true at search start but `Threads != 1`, close the sidecar immediately.
- Note for callers: because `session` snapshots `Threads`, `Hash`, and `MultiPV` when `TraceLog` is set, the collector always sends `Threads`, `Hash`, and `MultiPV` before `TraceLog`.

### 2.2 Configured width vs. effective width

Distinguish two widths:

- **Configured width** = the UCI `MultiPV` option value at the moment `TraceLog` is opened (`session.options.multipv`).
  - Controls the line schema: `schema = 3` when configured width is `1`; `schema = 4` when configured width `> 1`.
  - Controls whether `attempt` events carry the `"line"` field (present iff configured width `> 1`).
- **Effective width** = `min(configured MultiPV, rootMoves.size())` computed in `iterative_deepening` (`search_start.multi_pv`).
  - Even if a root has only 1 legal move while configured `MultiPV` is 3, `schema` is still `4` and `attempt.line` is still emitted (`0`), while `search_start.multi_pv` and ` len(round_end.rank)` are `1`.

### 2.3 Call roles and recorded-node tree

Maintain per-thread trace state across `ss->ply`:

- `path`: stack of UCI move strings for the active recorded path from the root (`ply == path.size()`).
- `nodeRecorded[ply]`: boolean marking whether the node at `ply` is on the recorded tree.
- `suppressDepth`: counter hiding re-entrant searches of the same node.

Classify every call into `search<nodeType>` through a wrapper `searchTraced<nodeType>(..., role, move, order)` with one of seven roles:

| Role | Stockfish call site in `search.cpp` | Writes `candidate`? | Opens recorded child (`nodeRecorded[ply] = true`)? |
|---|---|---|---|
| `Root` | Root aspiration call `search<Root>` in `iterative_deepening` | No | Root (`ply = 0`) is recorded (`nodeRecorded[0] = true`) |
| `First` | First move in the move loop (`moveCount == 1`): both the `NonPV` full-depth search (when LMR is skipped on a non-PV node) and the `PV` full-window search | Yes, if parent (`ply - 1`) is recorded | Yes, if parent is recorded |
| `Promote` | `PvNode` full-window search after a null-window fail-high (`moveCount > 1 && value > alpha`) | Yes, if parent is recorded | Yes, if parent is recorded |
| `Probe` | Reduced LMR null-window search (`depth >= 2 && moveCount > 1`), or full-depth non-PV search when LMR is skipped and `moveCount > 1` | Yes, if parent is recorded | **No** (subtree is not recorded) |
| `Rescore` | Full-depth null-window re-search after LMR fail-high (`value > alpha && d < newDepth` and `newDepth > d`) | Yes, if parent is recorded | **No** (subtree is not recorded) |
| `Auxiliary` | Null-move pruning search and ProbCut `search<NonPV>` | No | No (`nodeRecorded[ply] = false`) |
| `Suppress` | Null-move verification search and singular-extension search (re-entering at the same `ss`) | No | Increments `suppressDepth` for the whole call so nothing inside logs |

Key rules:

- Do **not** decide whether a child is recorded by checking `beta - alpha == 1`. A `NonPV` node's first move (`moveCount == 1` when LMR is skipped) uses a null window `(-(alpha+1), -alpha)` with role `First`, and **does** open a recorded child if its parent is recorded. Always key off `role in {First, Promote}` and `parentRecorded`.
- When `parentRecorded` is true for `First`, `Probe`, `Rescore`, or `Promote`, push the UCI string of `move` onto `path` before recursing, and pop it after writing `candidate`.
- `order` is the parent move-loop's 1-based `moveCount` for that move. If the same move is searched by `Probe`, then `Rescore`, then `Promote`, all three `candidate` events share the same `order`. Pruned moves still advance `moveCount`, so `order` can have gaps.
- When `openChild && depth <= 0` (the child `search` will drop straight into `qsearch` at entry), arm a `frontier` flag for that child node. Inside `search` when `depth <= 0`: run `qsearch`, and if `qsearch` did **not** take an immediate early exit (upcoming repetition cutoff at top of `qsearch`, or immediate `pos.is_draw(ss->ply) || ss->ply >= MAX_PLY`), write a `frontier` event before returning. Its `raw`, `cp`, `wdl`, and `bound` must equal the `candidate` event that opened it.

### 2.4 Hook locations in Stockfish 17.1

1. **`UCIEngine::position` (`uci.cpp`)**: after `engine.set_position(fen, moves)`, trim trailing whitespace from the reconstructed `fen` string (tokens after `position fen` joined by single spaces, or `StartFEN` for `startpos`) and call `Trace::position(fen)`.
2. **`Search::Worker::start_searching` (`search.cpp`)**:
   - At entry on main thread: if `Trace::active()` and `Threads != 1`, close the sidecar.
   - At search end on main thread, right before `updates.onBestmove(bestmove, ponder)`: call `Trace::end_search(bestmove, bestMove != Move::none(), ponder, !ponder.empty(), threads.nodes_searched(), bestThread->selDepth, int(bestThread->completedDepth))`.
3. **`Search::Worker::iterative_deepening` (`search.cpp`)**:
   - After `multiPV = std::min(multiPV, rootMoves.size())`, on main thread: call `Trace::search_start(limits.depth, int(multiPV), nodes)`.
   - At the top of each `rootDepth` iteration, initialize `int traceAttempt = 0;` (resets to 0 at every depth round, for both `W = 1` and `W > 1`).
   - Inside the `for (pvIdx = 0; pvIdx < multiPV; ++pvIdx)` loop, after setting the initial aspiration `alpha`/`beta` and `optimism`, if `mainThread && pvIdx == 0 && Trace::active()`: collect `rootOrder` (the UCI strings of `rm.pv[0]` for every `rm` in `rootMoves`) and call `Trace::begin_round(rootDepth, rootDepth, alpha, beta, rootOrder)`.
   - Inside the aspiration `while (true)` loop:
     - Before `searchTraced<Root>`: call `Trace::begin_attempt(rootDepth, ++traceAttempt, int(pvIdx), alpha, beta)`. Note that for `W > 1`, `traceAttempt` keeps incrementing across `pvIdx = 0 .. multiPV - 1` within the same round.
     - Immediately after `searchTraced<Root>` returns `bestValue`: call `Trace::attempt_result(bestValue, ...)`.
   - After `std::stable_sort(rootMoves.begin() + pvFirst, rootMoves.begin() + pvIdx + 1)`, when `mainThread && (pvIdx + 1 == multiPV || threads.stop) && Trace::active()`: build `rank` of length `multiPV` from `rootMoves[0 .. multiPV - 1]` (each entry's `move = rm.pv[0]`, `raw = rm.score`, `cp`, `wdl`, and full `pv`) and call `Trace::end_round(rootDepth, rootDepth, traceAttempt, rank, bestMoveChanges, selDepth, nodes)`. Always run traced searches with fixed `go depth D` and never abort mid-round with `stop`.

## 3. Exact JSONL sidecar specification

Format rules enforced by `src/pondera/data/trace.py`:

- UTF-8 JSON Lines, one JSON object per line, `\n` line endings, no blank lines, no floats anywhere (all numbers are integers; `parse_float` is rejected).
- Every event starts with `"schema": 3` (if configured `MultiPV == 1`) or `"schema": 4` (if configured `MultiPV > 1`), `"seq": <1-based line number>`, `"type": "<event_type>"`.
- Exactly one `session` (must be `seq = 1`), one `position`, one `search_start`, and one `search_end` (must be the last line).
- Score conventions:
  - Ordinary score iff `abs(raw) < value_mate - max_ply`.
  - When `raw` is ordinary, the event **must** include integer `"cp": UCIEngine::to_cp(raw, pos)` and `"wdl": [win, draw, loss]` (`UCIEngine::wdl_values(raw, pos)`, three integers in `0..1000` summing to `1000`).
  - When `raw` is a mate score (`abs(raw) >= value_mate - max_ply`), the event **must omit** both `"cp"` and `"wdl"`.
  - Perspective: `attempt.raw` and `round_end.rank[*].raw` are from the root side to move. `candidate.raw` and `frontier.raw` are from the **child** node's side to move (the `value` returned inside the child's `search` before the parent negates it), and their `"window": {"alpha": alpha, "beta": beta}` is the window passed into that child `search`.
  - `"bound"` is computed from the event's own `(raw, alpha, beta)`: `"upper"` if `raw <= alpha`, `"lower"` if `raw >= beta`, else `"exact"`.

Event payloads (in addition to `schema`, `seq`, `type`):

1. **`session`**:
   `{"engine": "<engine_version_info()>", "value_mate": 32000, "max_ply": 246, "options": {"threads": 1, "hash_mb": <int>, "multipv": <configured_multipv>}}`
2. **`position`**:
   `{"fen": "<canonical_fen>"}` (must be a legal non-terminal FEN).
3. **`search_start`**:
   `{"depth": <limits.depth>, "multi_pv": <effective_width>}` where `effective_width = min(configured_multipv, legal_root_moves)`.
4. **`round_start`**:
   `{"round": <d>, "depth": <d>, "window": {"alpha": <int>, "beta": <int>}, "root_order": ["<uci>", ...]}` (`round == depth`, contiguous `1..D`).
5. **`attempt`**:
   - Schema 3 (`configured_multipv == 1`):
     `{"round": <d>, "attempt": <a>, "window": {"alpha": <int>, "beta": <int>}, "raw": <int>, ["cp": <int>, "wdl": [w,d,l],] "bound": "<exact|upper|lower>"}`
   - Schema 4 (`configured_multipv > 1`): adds `"line": <0..effective_width-1>` right after `"window"`.
   - Within each round, `attempt` numbers are `1..N`. Each `line` in `0..effective_width-1` must end its attempt sequence with its single `"exact"` attempt.
6. **`candidate`**:
   `{"round": <d>, "attempt": <a>, "path": ["<uci>", ...], "ply": <len(path)>, "move": "<path[-1]>", "order": <1-based moveCount>, "call": "<first|probe|rescore|promote>", "depth": <child_depth_arg>, "window": {"alpha": <int>, "beta": <int>}, "raw": <int>, ["cp": <int>, "wdl": [w,d,l],] "bound": "<exact|upper|lower>", "nodes_delta": <uint>, "seldepth": <int>[, "pv": ["<uci>", ...]]}`
   - `"pv"` is present **if and only if** `ply == 1` and `call in {"first", "promote"}`. It starts with `move`, is legal from the root, and must not continue past a rule terminal (`PathTracker.search_terminal()`).
   - For `call in {"probe", "rescore"}`, `beta - alpha` must equal `1`.
   - At `ply == 1` in an accepted attempt, any move whose last call is a `probe`/`rescore` with `bound == "upper"` (meaning child `raw <= alpha`, so parent `-raw >= -alpha`, a root null-window fail-high) must also have an exact `first`/`promote` call in the same attempt.
7. **`frontier`**:
   `{"round": <d>, "attempt": <a>, "path": ["<uci>", ...], "ply": <len(path)>, "raw": <int>, ["cp": <int>, "wdl": [w,d,l],] "bound": "<exact|upper|lower>", "nodes_delta": <uint>, "seldepth": <int>}`
   - At most one `frontier` per `(round, attempt, path)`, and its `raw`, `cp`, `wdl`, and `bound` must match a `first` or `promote` candidate at the same `(round, attempt, path)`.
8. **`round_end`**:
   `{"round": <d>, "depth": <d>, "attempt": <total_attempts_in_round>, "rank": [{"move": "<uci>", "raw": <int>, ["cp": <int>, "wdl": [w,d,l],] "pv": ["<uci>", ...]}, ...], "counters": {"best_move_changes": <int>, "seldepth": <int>, "nodes": <nodes_since_search_start>}}`
   - `len(rank) == effective_width`, sorted in non-increasing order of `raw`.
   - Every rank entry must match an establishing `first`/`promote` root candidate (`ply == 1`, `bound == "exact"`, `candidate.raw == -entry.raw`, `candidate.pv == entry.pv`) from an accepted attempt of that round.
9. **`search_end`**:
   `{"bestmove": "<uci>"|null, "ponder": "<uci>"|null, "nodes": <uint>, "seldepth": <int>, "depth": <deepest_completed_depth>}`
   - `bestmove` must equal `round_ends[-1].rank[0].move`.

Generate a unified diff between the unmodified Stockfish 17.1 directory and your instrumented treatment directory and save it as `instrumentation.patch` in the external workspace.

## 4. Consistency gates

Before producing a dataset, verify on test roots covering ordinary midgames, mates, constrained-legal-move roots, and both `MultiPV = 1` and `MultiPV > 1`:

1. **Handshake diff**: between `uci` and `uciok`, the only stdout difference between control and treatment is the single `option name TraceLog type string default` line.
2. **Zero search perturbation**: after `uciok`, for identical UCI commands (`Threads=1`, fixed `Hash`, fixed `MultiPV`, `position fen ...`, `go depth D`), stdout of control, treatment with `TraceLog` empty, and treatment with `TraceLog` set must be byte-identical after stripping only wall-clock `time <int>` and `nps <int>` tokens from `info` lines.
3. **Determinism**: two fresh treatment processes with the same commands produce byte-identical sidecar files.
4. **Consumer validation**: every test sidecar passes `parse_sidecar(lines, teacher_width=W)` in `src/pondera/data/trace.py`; `synthesize_root` and `replay` accept the resulting traces.

If any gate fails, stop and fix the C++ instrumentation.

## 5. Connecting to Pondera's pipeline

1. **Configure paths**: edit a working copy of `configs/trace.toml`. Set `[paths].engine` to the treatment binary, `output` to the release directory, and `archive`/`roots` for automatic root-pool construction. If `roots` already exists, its JSONL records need a unique `root_id`, `split` (`train` or `val`), a legal non-terminal six-field `fen`, and `repetition` in `1..4`; `turn` and `position_key` are checked when present.
2. **Configure production**: set the top-level `seed`, `[roots]` sizes and claimable share, `[search].hash_mb` and `joint`, `[synthesis]` BOARD/draw/presentation settings, and `[release]` shard size and workers. `joint` is an array of `{width, depth, probability}` tables: widths are positive integers or `"full"`, depths are positive integers, pairs are unique, and probabilities sum to one. Each root draws a single pair; `"full"` resolves to its legal root-move count before UCI setup. Root rows record concrete `teacher_width` and capped `effective_width`. Presentation is drawn independently and accepts one value or a probability table. There is no separate recipe, smoke, collection, or release configuration.
3. **Run the single entry point**:
   ```bash
   uv run python -m pondera.data.pipeline --config /abs/path/to/trace.toml
   ```
   The pipeline builds or loads the root pool, runs each root in a fresh engine process (`Threads` -> `Hash` -> `MultiPV` -> `TraceLog` -> `position fen` -> `go depth`), validates and synthesizes each sidecar in memory, replays the selected trace, and atomically publishes shards. Sidecars exist only in temporary directories. `--start-shard` and `--end-shard` optionally select a shard range for job scheduling; `--workers` overrides worker count without changing trace semantics.
4. **Independently validate the completed release**:
   ```bash
   uv run python - <<'PY'
   from pondera.data.pipeline import validate_release
   validate_release("/abs/path/to/release")
   PY
   ```
   Validation checks manifests, array layout, stored states, and every trace replay against python-chess.
