# Teacher search records

A teacher record is the complete log of one instrumented Stockfish 17.1 search over one root position. Offline synthesis reads the record and writes trace tokens; the training pipeline never runs the engine. The instrumented build adds one string UCI option, `TraceLog`, which writes a JSONL sidecar. The change is observational: with tracing off the build is indistinguishable from the unmodified engine, and with tracing on the sidecar is the only addition.

This document is the contract between that build and the translator. It specifies where to observe the search, which events to write, and what every field means. The schema is versioned; the procedure and the schema change together. A release manifest records the SHA-256 of this document, so a record is interpretable only against the contract that produced it.

Stockfish is licensed under GPL-3.0-or-later. The instrumented source, the binary, and the records are external artifacts; this repository contains none of them. The procedure below describes observation points in prose and contains no engine source code.

## Instrumentation

### Fixed conditions

- Use the pinned Stockfish 17.1 source. Build a control binary from the unmodified source and a treatment binary from the instrumented source with the same compiler, flags, and embedded network.
- The collector runs one engine process per root. It sets `Threads`, `Hash`, and `MultiPV` first, then `TraceLog`, then sends `position fen <root>`, then `go depth D`, reads `bestmove`, and quits. It never reuses a process for another root, never supplies move history, and never uses time, node, or `searchmoves` limits.
- Tracing is defined only for `Threads = 1`. With any other thread count, ignore `TraceLog` and create no file.
- The instrumentation only observes. It must not change the position, evaluation, move ordering, transposition table, history tables, counters, or UCI output. Do not evaluate positions or render SAN for logging; write UCI moves from the move encoding.

### The option

Register a string UCI option named `TraceLog` with an empty default (off). Setting a nonempty path opens that file for truncation and writes the `session` event; setting an empty value closes it. Read the option only between searches and ignore changes during a search. If the file cannot be opened, leave tracing off and write nothing to stdout or stderr; the collector detects the missing file.

The `session` event records the engine's UCI name, the build's mate constant and maximum ply, and the effective `Threads`, `Hash`, and `MultiPV`. The constants let consumers classify mate scores without hard-coding a build.

### Write points

1. **Position.** After the `position fen` command has been applied, record the FEN exactly as supplied in the command, not a re-rendered position. The collector supplies the root FEN only, with no moves and no history.
2. **Search start.** When `go` starts, record the requested depth and the effective MultiPV.
3. **Round start.** At the top of each depth iteration, after the root moves have been ordered for the new depth and before the first aspiration pass, record the 1-based round number, the depth, the opening aspiration window, and the ordered root moves as UCI. The order is the root move list order; the transposition-table move is first when present.
4. **Attempts.** Each root search pass is one aspiration attempt. After a pass returns, record the 1-based attempt number within the round, the exact window passed to the root search, the returned root value, and its exact-or-bound classification. Record failed passes; never delete or rewrite them. The attempt number is part of every event produced during that pass.
5. **Round end.** After the aspiration loop accepts a pass, record the round, the accepted attempt number, the top W root moves with their values and principal variations, and the round counters.
6. **Search end.** Where the engine emits `bestmove`, record the move, the ponder move, total nodes, the engine's seldepth counter, and the completed depth.

### Move path and call roles

Maintain a stack of UCI moves in the searching thread. Push a move before the child search and pop it after the call returns; clear the stack when a search starts. The quiescence search does not push, because capture sequences inside it are not path moves.

Classify every call to the search entry with exactly one role:

| Role | Meaning | Recorded |
|---|---|---|
| `first` | First candidate at a node, searched with the full window. | Yes, as a candidate event. |
| `probe` | Null-window test of a later candidate, possibly at reduced depth. | Yes, as a candidate event. |
| `rescore` | Null-window re-test at full depth after a probe fails high. | Yes, as a candidate event. |
| `promote` | Full-window re-search after a null-window fail-high. | Yes, as a candidate event. |
| `iid` | Internal iterative deepening. | No; never opens a recorded child, even when its window is full. |
| `singular` | Singular-extension verification. | No; never opens a recorded child. |

If a call site fits none of these roles, stop and report it; do not invent a role.

The root is recorded. A `first` or `promote` call from a recorded node opens the child for recording. If that call later returns a bound rather than an exact value, the child's events stay in the file but are not used during translation.

A recorded node writes one candidate event for each of its `first`, `probe`, `rescore`, and `promote` calls. It writes nothing for `iid`, `singular`, or for candidates that are pruned before any search call. A null-window test writes exactly one line at the node that issues it and never records the tested child's subtree; this is the cheap-test rule.

A candidate event carries the child path (root to child, ending with the tested move), the parent's candidate order, the role, the child's depth argument, the window passed to the child, the returned raw value and bound, the node-count delta, and the child's seldepth. A root full-window candidate, meaning a `first` or `promote` call whose path has length one, also carries the principal variation.

A frontier event is written when a recorded node's `first` or `promote` child call reaches depth zero and its value comes from the quiescence search. Write it after the quiescence search returns, with the child path, the quiescence value and bound, the node-count delta, and seldepth. Do not record anything inside the quiescence search. A rule draw detected before the quiescence search has no frontier event; its value is carried by the parent's candidate event. The frontier event's raw value must equal the candidate event's raw value for the same call; the duplicate is intentional and is used for validation.

Candidate `order` is the 1-based position of the candidate in the parent's move-loop sequence for that attempt. Every call that tests the same candidate shares the order. It is not the count of logged events, so pruned moves leave gaps. At the root it is the actual test order; `round_start.root_order` is the order at round entry.

Events are written when the observed work finishes, so a child's event precedes its parent's candidate event. Line order is completion order, not narration order; reconstruct the tree from `path` and `ply`, not from adjacency.

### Determinism

- Write integer fields only. Do not write clocks, timestamps, thread or process identifiers, addresses, floating-point numbers, or locale-dependent formatting.
- Use a fixed key order, compact JSON separators, one object per line, and LF line endings.
- Flush each line before returning from the write point.
- Running the same command twice in fresh processes with the same binary and options must produce byte-identical stdout and byte-identical sidecars.
- Do not write the sidecar path, file offsets, host names, or environment values.

### Consistency checks

The gate before any collection. Every comparison starts with `uci` and compares the bytes produced after `uciok`: the treatment build advertises `TraceLog` in its handshake, and that one added option line is the only permitted handshake difference. From the option and position commands onward, the output must be byte-identical.

1. Control binary with tracing off versus treatment binary with tracing off: identical post-handshake output, no sidecar.
2. Treatment binary with tracing on versus treatment binary with tracing off: identical post-handshake output; the sidecar is the only new output.
3. Treatment binary with tracing on, run twice with a fresh process and an empty hash table each time: identical post-handshake output and identical sidecar bytes.
4. Official Stockfish 17.1 binary versus treatment binary with tracing on, using the same commands and omitting `TraceLog` for the official run: identical post-handshake output. Any difference stops the build; do not collect from it and do not explain the difference away.
5. Structural check of the sidecar: every line parses, `seq` runs from 1 to the line count, there is one root, each round has one accepted attempt, and the invariants below hold.

A minimal command sequence is:

```text
uci
setoption name Threads value 1
setoption name Hash value 256
setoption name MultiPV value 1
setoption name TraceLog value sidecar.jsonl
position fen <root fen>
go depth 16
quit
```

The traced run adds the `TraceLog` line; the control and official runs omit it. The handshake through `uciok` is excluded from the byte comparisons; it differs by the added option advertisement.

## Record schema

### File and line format

- UTF-8 JSON Lines. One object per line, no blank lines, compact separators, fixed key order, LF line endings.
- Every line carries `schema` (currently 1), `seq` (the 1-based line number), and `type`.
- One file holds one root: exactly one `session`, one `position`, one `search_start`, and one `search_end`. `session` is the first line and `search_end` is the last.
- A consumer rejects a different schema version or a non-monotonic `seq`.
- The file contains no floating-point numbers.

### Scores, windows, and bounds

- `raw` is an integer in engine units. A normal score is in centipawns. A mate score for the side to move is `value_mate - ply_to_mate`; a mated score is `-value_mate + ply_to_mate`. The distance unit is a ply. A value is a mate score when its magnitude is at least `value_mate - max_ply`; both constants are in the `session` event.
- `raw` is from the side to move at the node the event describes. The root player's perspective is `raw * (-1) ** ply`, where `ply` is the node's distance from the root. `attempt` and `round_end` values are already at the root (ply zero).
- `window` is the exact `{alpha, beta}` pair passed to the call, in the frame of the node being searched. `bound` is `exact` when `alpha < raw < beta`, `upper` when `raw <= alpha`, and `lower` when `raw >= beta`. A null window cannot produce `exact`.
- Infinite sentinels may appear in windows and in failed-attempt results. They never appear as the value of a completed round.
- The translator converts mate distances from plies to moves. The engine record never performs that conversion.

### Paths and moves

- Moves are lowercase UCI strings in the absolute board frame. They are never rank-reflected; the translator mirrors Black-to-move roots when it writes tokens.
- `path` is the sequence of moves from the root to the node the event describes. Candidate and frontier events describe the child, so `path` ends with the tested move and `ply` equals the path length. Root-scope events have no `path`.
- A principal variation is a UCI list. A root candidate's variation starts with that candidate; a ranked variation starts with its root move.

### Events

Every event also carries `schema`, `seq`, and `type`.

**`session`**

| Field | Type | Meaning |
|---|---|---|
| `engine` | string | The engine's UCI name. |
| `value_mate` | integer | The build's mate constant. |
| `max_ply` | integer | The build's maximum search ply. |
| `options` | object | Effective `threads`, `hash_mb`, and `multipv` when tracing opened. |

**`position`**

| Field | Type | Meaning |
|---|---|---|
| `fen` | string | The root FEN supplied in the `position fen` command, verbatim. |

**`search_start`**

| Field | Type | Meaning |
|---|---|---|
| `depth` | integer | Requested search depth. |
| `multi_pv` | integer | Effective MultiPV. |

**`round_start`**

| Field | Type | Meaning |
|---|---|---|
| `round` | integer | 1-based depth iteration. |
| `depth` | integer | Current iteration depth. |
| `window` | object | Opening aspiration window `{alpha, beta}` at the root, root perspective. |
| `root_order` | array | Ordered root moves as UCI at round entry. |

**`attempt`**

| Field | Type | Meaning |
|---|---|---|
| `round` | integer | Round number. |
| `attempt` | integer | 1-based aspiration pass within the round. |
| `window` | object | Window passed to the root search, root perspective. |
| `raw` | integer | Returned root value, root perspective. |
| `bound` | string | `exact`, `lower`, or `upper` relative to `window`. |

**`candidate`**

| Field | Type | Meaning |
|---|---|---|
| `round` | integer | Round number. |
| `attempt` | integer | Attempt number. |
| `path` | array | Root-to-child UCI moves, ending with `move`. |
| `ply` | integer | Path length. |
| `move` | string | The tested move; the last path element. |
| `order` | integer | 1-based candidate order in the parent's move loop for this attempt, shared by all calls for the candidate. |
| `call` | string | `first`, `probe`, `rescore`, or `promote`. |
| `depth` | integer | Depth argument received by the child search. |
| `window` | object | Window passed to the child, child perspective. |
| `raw` | integer | Returned child value, child perspective. |
| `bound` | string | `exact`, `lower`, or `upper` relative to `window`. |
| `nodes_delta` | integer | Nodes consumed by the call. |
| `seldepth` | integer | The child's seldepth counter at return. |
| `pv` | array | Principal variation; required when `ply` is one and `call` is `first` or `promote`; omitted otherwise. |

**`frontier`**

| Field | Type | Meaning |
|---|---|---|
| `round` | integer | Round number. |
| `attempt` | integer | Attempt number. |
| `path` | array | Root-to-frontier-node UCI moves. |
| `ply` | integer | Path length. |
| `raw` | integer | Quiescence-search return value, frontier-node perspective. |
| `bound` | string | `exact`, `lower`, or `upper` relative to the window passed to the child. |
| `nodes_delta` | integer | Nodes consumed by the quiescence search. |
| `seldepth` | integer | Seldepth counter at return. |

**`round_end`**

| Field | Type | Meaning |
|---|---|---|
| `round` | integer | Round number. |
| `depth` | integer | Depth of the completed round. |
| `attempt` | integer | The accepted attempt number. |
| `rank` | array | Up to W entries `{move, raw, pv}`, sorted by `raw` descending, ties in root order. `raw` is root perspective; `pv` starts with `move`. |
| `counters` | object | `best_move_changes` (the engine's root best-move change counter for the round), `seldepth` (the engine's counter at round end), and `nodes` (cumulative nodes since search start). |

**`search_end`**

| Field | Type | Meaning |
|---|---|---|
| `bestmove` | string or null | The move emitted as `bestmove`; null when the engine has none. |
| `ponder` | string or null | The ponder move, when present. |
| `nodes` | integer | Total nodes searched. |
| `seldepth` | integer | The engine's seldepth counter at the end. |
| `depth` | integer | Deepest completed depth. |

### Invariants

- `seq` equals the line number. `session` is first and `search_end` is last. There is exactly one `position`, one `search_start`, and one root per file.
- `session.options.multipv` and `search_start.multi_pv` equal the collection's teacher width.
- Rounds are numbered 1 through the deepest completed depth without gaps. Every `candidate` and `frontier` references an attempt of its round. No event with a given round appears after that round's `round_end`.
- For the supported width of one, exactly one attempt per round has `bound = exact`; `round_end.attempt` names it. Events of other attempts are present and are not used by the translator.
- In an accepted attempt, every full-window call at a recorded node returns `exact`. A full-window call that returns a bound is a contract violation; synthesis must stop rather than coerce it.
- A null-window fail-low is the only evidence for a refutation. A null-window fail-high is not a statement by itself; it must be followed by a deeper null-window test or a full-window `promote`.
- All calls that test the same candidate in one attempt share the same `order`.
- When a frontier event exists, its `raw` equals the `raw` of the corresponding candidate event.
- In an accepted attempt, `round_end.rank[0].raw` equals `attempt.raw`. `rank` is sorted by `raw` descending.
- `search_end.bestmove` equals `rank[0].move` of the deepest `round_end`; `ponder`, when present, is the second move of that variation.
- Principal variations are legal from the root and stop at a rule terminal. The record is raw search evidence: it is not rank-reflected and is not a token trace.
- No field contains a clock, a thread or process identifier, an address, a floating-point number, or a locale-dependent value.

### Root metadata

Each sidecar is accompanied by one root metadata record. The fields are outside the sidecar because the engine never receives them:

| Field | Type | Meaning |
|---|---|---|
| `root_id` | string | Stable identifier of the root in the source release. |
| `split` | string | `train` or `val`. |
| `fen` | string | The FEN supplied to the engine; equals `position.fen`. |
| `repetition` | integer | Occurrences of the root position in the source game up to and including this occurrence; at least one; roots with five or more are excluded. |
| `provenance` | object | Source release hash, source row key, and any fields needed to reproduce the root. |

The engine sees only `fen`. The sidecar contains no game-history field. Replay starts from `fen` and counts repetitions along the written path only; it never uses `repetition`. The source-game `repetition` is used for the board input atom and for draw-claim eligibility.

A root is claimable when `repetition` is at least three or the FEN halfmove clock is at least 100. If it is claimable and the champion's root value from the deepest `round_end` is at most zero, the label action is a draw claim with no variation; otherwise the label action is the champion move with its normal variation. The value written with a claim is the root evaluation, not an action value, so the calibration is "claim exactly when the reported evaluation is at most zero". Roots with no legal move, checkmate, stalemate, or repetition of five or more are excluded by root selection.

### Rule terminals

Rule terminals are not separate events. Checkmate, stalemate, repetition, and the 50-move rule appear as ordinary node values: a mate score for a mate and zero for a draw. The translator reproduces them from the root FEN and the written path. Repetition follows the engine's rule: a position that occurred once strictly after the root, or at least twice anywhere in the search path including the root, is terminal; a return to the root is not terminal on its second occurrence, only on its third. When checkmate and the 50-move rule coincide, checkmate wins.

### Use by the translator

- Take the accepted attempt of each round. A full-window exact call is a completed comparison; its child's records are the evidence for that value.
- A null-window fail-low is a refutation and carries no value. A null-window fail-high is replaced by the full-window re-search that follows it.
- Pruned candidates and auxiliary searches have no events and are not represented. Failed attempts are evidence that the search retried, not part of the demonstrated line.
- The deepest round's `rank[0]` supplies the final move, evaluation, and variation. The root's evaluation is `attempt.raw` of the deepest accepted attempt.

### Manifest

The collection manifest records, at minimum:

- the record schema version and the SHA-256 of this document;
- the SHA-256 of the collection configuration;
- the engine name, the treatment binary's SHA-256, the instrumentation diff's SHA-256, and the control binary's identity;
- the effective search options: depth, teacher width, threads, hash size, and the fixed options;
- the root selection: source release hash, position key, seed, sample counts, and split counts;
- the presentation, board, and synthesis parameters;
- the output file hashes and row counts added by the release builder.

Sidecars and root metadata stay with the run; they are not part of a training release.

### What is not recorded

The record does not contain transposition-table hits or reuse, pruning and futility decisions, reduction amounts beyond the call role and depth argument, quiescence-search internals, static evaluations before the quiescence search, auxiliary searches, individual events inside unrecorded subtrees, wall-clock time, memory use, or thread scheduling. Absence is part of the record: an untested candidate has no event, a pruned subtree has no record, and a failed attempt is present but tagged.
