# Teacher search records

A teacher record is the complete log of one instrumented Stockfish 17.1 search over one root position. Offline synthesis reads the record and writes trace tokens; the training pipeline never runs the engine. The instrumented build adds one string UCI option, `TraceLog`, which writes a JSONL sidecar. The change is observational: with tracing off the build is indistinguishable from the unmodified engine, and with tracing on the sidecar is the only addition.

This document is the contract between that build and the translator. It specifies where to observe the search, which events to write, and what every field means. The schema is versioned; the procedure and the schema change together. A release manifest records the schema version and the SHA-256 of this document, so a record is interpretable only against the contract that produced it.

Stockfish is licensed under GPL-3.0-or-later. The instrumented source, the binary, and the records are external artifacts; this repository contains none of them. The procedure below describes observation points in prose and contains no engine source code.

## Instrumentation

### Fixed conditions

- Use the pinned Stockfish 17.1 source. Build a control binary from the unmodified source and a treatment binary from the instrumented source with the same compiler, flags, and embedded network.
- The collector runs one engine process per root. It sets `Threads`, `Hash`, and `MultiPV` first, then `TraceLog`, then sends `position fen <root>`, then `go depth D`, reads `bestmove`, and quits. It never reuses a process for another root, never supplies move history, and never uses time, node, or `searchmoves` limits.
- Tracing is defined only for `Threads = 1`; with any other value, ignore `TraceLog` and create no file. The teacher width W (MultiPV) selects the schema: W = 1 emits schema 3, and W > 1 emits schema 4, which numbers attempts across the whole round, tags each attempt with its PV line, and ends a round after the last line's pass. The schema below is stated for teacher width one except where the width section says otherwise.
- The instrumentation only observes. It must not change the position, evaluation, move ordering, transposition table, history tables, counters, or UCI output. Do not evaluate positions or render SAN for logging; write UCI moves from the move encoding.

### The option

Register a string UCI option named `TraceLog` with an empty default (off). Setting a nonempty path opens that file for truncation and writes the `session` event; setting an empty value closes it. Read the option only between searches and ignore changes during a search. If the file cannot be opened, leave tracing off and write nothing to stdout or stderr; the collector detects the missing file.

The `session` event records the engine's UCI name, the build's mate constant and maximum ply, and the effective `Threads`, `Hash`, and `MultiPV`. The constants let consumers classify mate scores without hard-coding a build.

### Write points

1. **Position.** After the `position fen` command has been applied, record the FEN exactly as supplied in the command, not a re-rendered position. The recorded string joins the command's FEN tokens with single spaces; the collector supplies a canonical single-spaced FEN, so the record matches the command byte for byte. The collector supplies the root FEN only, with no moves and no history.
2. **Search start.** When `go` starts, record the requested depth and the effective MultiPV.
3. **Round start.** At the top of each depth iteration, after the root moves have been ordered for the new depth and before the first aspiration pass, record the 1-based round number, the iteration depth, the opening aspiration window, and the ordered root moves as UCI. The iteration depth is the value the engine reports as UCI `info depth`; it is not the possibly reduced depth of an aspiration re-search. A candidate event's `depth` carries the actual depth argument received by the child. The order is the root move list order; the transposition-table move is first when present.
4. **Attempts.** Each root search pass is one aspiration attempt. After a pass returns, record the attempt number, the exact window passed to the root search, the returned root value, and its exact-or-bound classification. Record failed passes; never delete or rewrite them. The attempt number is part of every event produced during that pass. With teacher width one the number restarts at one every round. With W > 1 the number keeps incrementing across the round's PV lines and the attempt event carries `line`, the 0-based PV line index; each PV line runs its own aspiration loop and its accepted pass is its last attempt, which is the attempt of that line with `bound = exact`.
5. **Round end.** With width one, after the aspiration loop accepts a pass. With W > 1, after the last PV line's aspiration loop accepts, when every rank entry carries a value from this round. Record the round, the round's total attempt count, the top W root moves with their values and principal variations, and the round counters.
6. **Search end.** Where the engine emits `bestmove`, record the move, the ponder move, total nodes, the engine's seldepth counter, and the completed depth.

### Move path and call roles

Maintain a stack of UCI moves in the searching thread. Push a move before the child search and pop it after the call returns; clear the stack when a search starts. The quiescence search does not push, because capture sequences inside it are not path moves.

Classify every call to the search entry with exactly one role:

| Role | Meaning | Recorded |
|---|---|---|
| `root` | The search of the root position. | Yes, as the root; it has no candidate event of its own, and its value is carried by the round's attempt and round-end events. |
| `first` | First candidate at a node, searched with the node's window: the full window at a PV node, the node's own null window at a non-PV node. | Yes, as a candidate event. |
| `probe` | Null-window test of a later candidate, possibly at reduced depth. | Yes, as a candidate event. |
| `rescore` | Null-window re-test at full depth after a probe fails high. | Yes, as a candidate event. |
| `promote` | Re-search with the node's window after a null-window fail-high; at a null-window PV node the node's window is itself a null window. | Yes, as a candidate event. |
| `auxiliary` | Auxiliary child search such as the null-move search or ProbCut. | No; never opens a recorded child. |
| `suppress` | Re-entrant search of the same node, such as null-move verification or singular-extension verification. | No; suppresses recording for the entire subtree. |

If a call site fits none of these roles, stop and report it; do not invent a role. Stockfish 17.1 has no internal iterative deepening (IID) call: it uses internal iterative reduction (IIR), which lowers the depth at the current node and never calls the search entry, so no IID role is needed.

The root is recorded. A `first` or `promote` call from a recorded node opens the child for recording. If that call later returns a bound rather than an exact value, the child's events stay in the file but are not used during translation.

A recorded node writes one candidate event for each of its `first`, `probe`, `rescore`, and `promote` calls. It writes nothing for `auxiliary` or `suppress` calls, or for candidates that are pruned before any search call. A null-window test writes exactly one line at the node that issues it and never records the tested child's subtree; this is the cheap-test rule.

A candidate event carries the child path (root to child, ending with the tested move), the parent's candidate order, the role, the child's depth argument, the window passed to the child, the returned `raw` value with its `cp` and `wdl` fields when the value is not a mate score, the bound, the node-count delta, and the child's seldepth. A root full-window candidate, meaning a `first` or `promote` call whose path has length one, also carries the principal variation.

A frontier event is written when a recorded node's `first` or `promote` child call reaches depth zero and its value comes from the quiescence search. Write it after the quiescence search returns, with the child path, the quiescence value, its `cp` and `wdl` when the value is not a mate score, the bound, the node-count delta, and seldepth. Do not record anything inside the quiescence search. A rule draw detected before the quiescence search has no frontier event; its value is carried by the parent's candidate event. The frontier event's `raw`, `cp`, and `wdl` values must equal the candidate event's for the same call; the duplicate is intentional and is used for validation.

Candidate `order` is the 1-based position of the candidate in the parent's move-loop sequence for that attempt. Every call that tests the same candidate shares the order. It is not the count of logged events, so pruned moves leave gaps. At the root it is the actual test order; `round_start.root_order` is the order at round entry.

Events are written when the observed work finishes, so a child's event precedes its parent's candidate event. Line order is completion order, not narration order; reconstruct the tree from `path` and `ply`, not from adjacency.

### Determinism

- Write integer fields only. Do not write clocks, timestamps, thread or process identifiers, addresses, floating-point numbers, or locale-dependent formatting.
- Use a fixed key order, compact JSON separators, one object per line, and LF line endings.
- Flush each line before returning from the write point.
- Running the same command twice in fresh processes with the same binary and options must produce byte-identical sidecars and stdout that is identical under the wall-clock rule stated in Consistency checks.
- Do not write the sidecar path, file offsets, host names, or environment values.

### Consistency checks

The gate before any collection. Every comparison starts with `uci` and compares the bytes produced after `uciok`: the treatment build advertises `TraceLog` in its handshake, and that one added option line is the only permitted handshake difference. From the option and position commands onward, the output must be byte-identical, with one exception for the engine's own wall-clock fields.

Stockfish prints a wall-clock `time` field and a derived `nps` field in every search info line. Those two values differ between any two processes, including two runs of the same official binary, so the comparison removes only their values; the field names, their order, and every other byte, including depth, seldepth, score, nodes, and the principal variation, remain compared. This exception is calibrated by also running the official binary twice and confirming that it differs only in those values. In the checks below, "identical" means identical under this rule.

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
- Every line carries `schema` (currently 3), `seq` (the 1-based line number), and `type`.
- One file holds one root: exactly one `session`, one `position`, one `search_start`, and one `search_end`. `session` is the first line and `search_end` is the last.
- A consumer rejects a different schema version or a non-monotonic `seq`.
- The file contains no floating-point numbers.

### Scores, windows, and bounds

- `raw` is an integer in the engine's internal Value units, not UCI centipawns. In Stockfish 17.1 a pawn is 208 internal units, and the engine converts an ordinary internal score to centipawns with a material-dependent scale before printing `score cp`; `raw` and the printed centipawns are therefore different numbers. A mate score for the side to move is `value_mate - ply_to_mate`; a mated score is `-value_mate + ply_to_mate`. The distance unit is a ply. A value is a mate score when its magnitude is at least `value_mate - max_ply`; both constants are in the `session` event.
- `cp` is the integer that the engine's own UCI conversion produces for `raw` at the position the event describes; for an ordinary score this is exactly the number printed as `score cp`. It is present exactly when `raw` is not a mate score. It is kept for analysis and readability; value quantization does not use it.
- `wdl` is the engine's win/draw/loss estimate for `raw` at the position the event describes: a three-integer array `[win, draw, loss]` in per-mille units that sums to 1000, produced by the same win-rate model that backs the engine's `UCI_ShowWDL` output. Enabling `UCI_ShowWDL` is not necessary and does not change the recorded values. It is present exactly when `raw` is not a mate score, and like `raw` it is from the side to move at the event's position, so candidate and frontier `wdl` are child-relative while attempt and rank `wdl` are root-relative.
- Value quantization uses the WDL-derived scalar `(wdl[0] - wdl[2]) / 1000`, which lies in [-1, 1]; this is the value the frozen 20-bin codec bins. Mate distances are converted from `raw` (plies) to moves. The collection uses no tablebases, so every non-mate value is an ordinary score.
- `raw` is from the side to move at the node the event describes. The root player's perspective is `raw * (-1) ** ply`, where `ply` is the node's distance from the root. `attempt` and `round_end` values are already at the root (ply zero).
- `window` is the exact `{alpha, beta}` pair passed to the call, in the frame of the node being searched. `bound` is `exact` when `alpha < raw < beta`, `upper` when `raw <= alpha`, and `lower` when `raw >= beta`. A null window cannot produce `exact`. Because `raw` is child-relative, an `upper` bound is a fail-low of the child and a fail-high of the move, while a `lower` bound is a fail-high of the child and a fail-low (refutation) of the move.
- Infinite sentinels may appear in windows and in failed-attempt results. They never appear as the value of a completed round.
- The translator converts mate distances from plies to moves. The engine record never performs that conversion.

### Paths and moves

- Moves are lowercase UCI strings in the absolute board frame. They are never rank-reflected; the translator mirrors Black-to-move roots when it writes tokens.
- `path` is the sequence of moves from the root to the node the event describes. Candidate and frontier events describe the child, so `path` ends with the tested move and `ply` equals the path length. Root-scope events have no `path`.
- A principal variation is a UCI list. A root candidate's variation starts with that candidate; a ranked variation starts with its root move. A variation may extend past the last written frontier into the quiescence search. Quiescence internals are not recorded; the frontier is the record boundary, and the tail is still part of the engine's variation.

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
| `depth` | integer | Iteration depth. This is the value the engine reports as UCI `info depth`; it is not the reduced depth of an aspiration re-search. |
| `window` | object | Opening aspiration window `{alpha, beta}` at the root, root perspective. |
| `root_order` | array | Ordered root moves as UCI at round entry. |

**`attempt`**

| Field | Type | Meaning |
|---|---|---|
| `round` | integer | Round number. |
| `attempt` | integer | 1-based aspiration pass within the round. |
| `window` | object | Window passed to the root search, root perspective. |
| `raw` | integer | Returned root value, root perspective. |
| `cp` | integer, omitted for mate scores | Centipawn value of `raw` at the root, using the engine's UCI conversion. |
| `wdl` | array of three integers, omitted for mate scores | Engine win/draw/loss estimate `[win, draw, loss]` at the root, per mille, summing to 1000. |
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
| `depth` | integer | Depth argument received by the child search; this is the actual child depth, which may differ from the round's iteration depth. |
| `window` | object | Window passed to the child, child perspective. |
| `raw` | integer | Returned child value, child perspective. |
| `cp` | integer, omitted for mate scores | Centipawn value of `raw` at the child position, using the engine's UCI conversion. |
| `wdl` | array of three integers, omitted for mate scores | Engine win/draw/loss estimate `[win, draw, loss]` at the child position, per mille, summing to 1000. |
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
| `cp` | integer, omitted for mate scores | Centipawn value of `raw` at the frontier node, using the engine's UCI conversion. |
| `wdl` | array of three integers, omitted for mate scores | Engine win/draw/loss estimate `[win, draw, loss]` at the frontier node, per mille, summing to 1000. |
| `bound` | string | `exact`, `lower`, or `upper` relative to the window passed to the child. |
| `nodes_delta` | integer | Nodes consumed by the quiescence search. |
| `seldepth` | integer | Seldepth counter at return. |

**`round_end`**

| Field | Type | Meaning |
|---|---|---|
| `round` | integer | Round number. |
| `depth` | integer | Iteration depth of the completed round. |
| `attempt` | integer | The accepted attempt number. |
| `rank` | array | Up to W entries `{move, raw, cp, wdl, pv}`, sorted by `raw` descending, ties in root order. `raw` is root perspective; `cp` and `wdl` are omitted for a mate entry; `pv` starts with `move`. |
| `counters` | object | `best_move_changes` (the engine's root best-move change counter for the round), `seldepth` (the engine's counter at round end), and `nodes` (cumulative nodes since search start). |

**`search_end`**

| Field | Type | Meaning |
|---|---|---|
| `bestmove` | string or null | The move emitted as `bestmove`; null when the engine has none. |
| `ponder` | string or null | The ponder move, when present. |
| `nodes` | integer | Total nodes searched. |
| `seldepth` | integer | The engine's seldepth counter at the end. |
| `depth` | integer | Deepest completed depth. |

### Teacher width greater than one (schema 4)

With W > 1 each depth iteration runs the PV lines in order. Line k searches the root moves from rank k onward; lines already established this round are skipped. Each line is a full aspiration loop with its own window, and the lines share one transposition table, so a later line's value for its remaining set can exceed an earlier line's value; the line index is not the final rank. The `round_end` rank is sorted after the round and is the authoritative order. A round with W lines therefore produces at least W distinct exact root candidates, one per line, before any refutation can be stated.

Every root move not established by an earlier line is called at least once in every pass of every later line; a tail candidate can be probed up to W times per round. A candidate's value for the round is its last exact full-window call across the round's accepted passes, and the only refutation evidence the translator uses is a null-window fail-low in the last line's accepted pass, where the window brackets the final W-th value. A candidate whose every last-line call is a full-window bound (a `first` or `promote` that failed outside the window) is omitted from the trace, exactly like an accepted-pass bound at width one.

### Invariants

- `seq` equals the line number. `session` is first and `search_end` is last. There is exactly one `position`, one `search_start`, and one root per file.
- `session.options.multipv` and `search_start.multi_pv` equal the collection's teacher width.
- Rounds are numbered 1 through the deepest completed depth without gaps. Every `candidate` and `frontier` references an attempt of its round. No event with a given round appears after that round's `round_end`.
- Exactly one attempt per PV line per round has `bound = exact`, and it is that line's last attempt. With width one, `round_end.attempt` names it; with W > 1, `round_end.attempt` is the round's total attempt count. Events of failed attempts are present and are not used by the translator.
- For each `rank` entry of a `round_end` there is, in the accepted pass of the line that established it, exactly one candidate at ply one whose call is `first` or `promote` with `bound = exact`, whose `raw` is the negation of the entry's `raw`, and whose `pv` equals the entry's `pv`; when a move earns exact calls in several lines, the establishing call is the last one in the round. A root call that establishes a rank value and returns a bound is a contract violation; synthesis must stop rather than coerce it. Full-window calls at internal recorded nodes may return a bound. That is normal PVS evidence, not a contract violation: the events stay in the file, the translator writes the child's records when they are used as evidence and writes no evaluation for that node, and synthesis does not stop.
- A null-window child `lower` bound is a fail-low of the move; it is the only evidence for a refutation and carries no value. A null-window child `upper` bound is a fail-high of the move and is not a statement by itself. A root fail-high is always followed by a deeper null-window test or a full-window `promote` within the same line's aspiration loop; a root fail-high without a follow-up is a contract violation. At an internal node the move may end in a beta cutoff with no follow-up; the candidate stays in the file and the translator does not represent it.
- All calls that test the same candidate in one attempt share the same `order`.
- When a frontier event exists, its `raw` equals the `raw` of the corresponding candidate event, and its `cp` and `wdl` equal the candidate's when present.
- In an accepted attempt, `round_end.rank[0].raw` equals `attempt.raw`, its `cp` equals `attempt.cp`, and its `wdl` equals `attempt.wdl` when present. `rank` is sorted by `raw` descending.
- `cp` is present exactly when `raw` is not a mate score; its value equals the engine's UCI conversion at the event's position.
- `wdl` is present exactly when `raw` is not a mate score; it is three nonnegative integers that sum to 1000 and equal the engine's win-rate model applied to `raw` at the event's position.
- Round `depth` values are iteration depths. A candidate's `depth` is the actual child depth argument and may differ.
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

A root is claimable when `repetition` is at least three or the FEN halfmove clock is at least 100. If it is claimable and the champion's root value from the deepest `round_end` is at most one, the label action is a draw claim with no variation; otherwise the label action is the champion move with its normal variation. The bound is one rather than zero because the engine's rule-draw score is `VALUE_DRAW - 1 + (nodes & 0x2)` (−1 or +1, an anti-cycling jitter): a claimable root whose best line is a draw reports +1 half the time, and the gate treats that whole band as "not better than a draw". The value written with a claim is the root evaluation, not an action value, so the calibration is "claim exactly when the reported evaluation is at most a draw". Roots with no legal move, checkmate, stalemate, or repetition of five or more are excluded by root selection.

### Rule terminals

Rule terminals are not separate events. Checkmate, stalemate, repetition, and the 50-move rule appear as ordinary node values: a mate score for a mate and zero for a draw. The engine's raw score for a rule draw is `VALUE_DRAW - 1 + (nodes & 0x2)` (−1 or +1), and a quiescence search does not detect stalemate, so a stalemated depth-zero leaf carries a static evaluation instead; the translator writes the exact zero value token for a rule draw regardless. The translator reproduces them from the root FEN and the written path. Repetition follows the engine's rule: a position that occurred once strictly after the root, or at least twice anywhere in the search path including the root, is terminal; a return to the root is not terminal on its second occurrence, only on its third. When checkmate and the 50-move rule coincide, checkmate wins.

### Use by the translator

- Take the accepted attempt of each round. With width W > 1, take the accepted attempt of each of the round's W lines instead: narrate each line in line order, writing each root candidate's evaluation once from the accepted pass of the line that produced its last exact call, then write the round's refutations after the W lines, using only null-window fail-lows from the last line's accepted pass. The root call that establishes the round's value is a full-window exact call; its child's records are the evidence for that value. The round's evaluation is the WDL-derived scalar `(rank[0].wdl[0] - rank[0].wdl[2]) / 1000` when `wdl` is present, and `rank[0].raw` (a mate score) otherwise. `rank[0].cp` remains available for analysis but is not quantized.
- Full-window calls at internal recorded nodes may return a bound; they are normal PVS evidence. The translator writes the child's records when it uses them and writes no evaluation for the node whose call is bounded.
- A null-window child `lower` bound (the move fails low) is a refutation and carries no value. A null-window child `upper` bound at the root is replaced by the full-window re-search that follows it; at an internal node a beta cutoff may leave it without a follow-up, and the candidate is not represented.
- Pruned candidates and auxiliary searches have no events and are not represented. Failed attempts are evidence that the search retried, not part of the demonstrated line.
- The deepest round's `rank[0]` supplies the final move, evaluation, and variation. The root's evaluation is `attempt.raw` with the WDL-derived scalar from `attempt.wdl` of the deepest accepted attempt; `attempt.cp` remains available for analysis.

### Manifest

The collection manifest records, at minimum:

- the record schema version and the SHA-256 of this document, which together identify the teacher contract; a revision of either changes that identity, and the manifest is regenerated with the contract;
- the SHA-256 of the collection configuration;
- the engine name, the treatment binary's SHA-256, the instrumentation diff's SHA-256, and the control binary's identity;
- the effective search options: depth, teacher width, threads, hash size, and the fixed options;
- the root selection: source release hash, position key, seed, sample counts, and split counts;
- the presentation, board, and synthesis parameters;
- the output file hashes and row counts added by the release builder.

Sidecars and root metadata stay with the run; they are not part of a training release.

### What is not recorded

The record does not contain transposition-table hits or reuse, pruning and futility decisions, reduction amounts beyond the call role and depth argument, quiescence-search internals, static evaluations before the quiescence search, auxiliary searches, individual events inside unrecorded subtrees, wall-clock time, memory use, or thread scheduling. Absence is part of the record: an untested candidate has no event, a pruned subtree has no record, and a failed attempt is present but tagged.
