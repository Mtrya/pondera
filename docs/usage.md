# Usage and data formats

Pondera is a library. Training and UCI serving have `python -m` entry points; offline data preparation and match evaluation are library functions that experiment run scripts compose. Paths are supplied explicitly or default to `.local/` beneath the working directory. Engine executables are resolved from `PATH` unless a path is supplied. The examples assume the training extra is installed.

## Data format

The board tokenizer produces `int16` arrays padded to the longest encoded position in each split. Each row is the set of occupied squares as self-contained atoms, `square * 12 + piece` in a1..h8 scan order, followed by eight metadata atoms: side to move, four castling rights, en passant, halfmove clock, and repetition count. Empty squares carry no token. Piece placement and the first four FEN fields round-trip exactly; the fullmove counter is not encoded, halfmove clocks above 100 share the top halfmove atom, and repetition counts above four share the top repetition atom. The input contract is FEN plus repetition count, not full game history. Padding is masked.

Move arrays contain `int16` indices into the 1,969-action vocabulary, including the historical draw-claim action. UCI emits board moves only and leaves draw adjudication to the match runner. Manifests include the tokenizer version and array checksums.

Beyond board encodings, `pondera.chess.vocabulary` freezes one shared id space (CHESS_TOKENIZER) for sequence models: 949 board and metadata atoms, 1,969 move tokens, 20 value-bin tokens, mate-distance tokens (win 0–28, loss 0–30; distance 0 is reserved), and eight trace delimiters (PAD, BACK, RESET, BOARD, FINAL, REFUTED, DEPTH, END), for 3,006 ids in total. Sequence models embed the whole space with one table; the board prefix carries no positional term, while trace tokens add sinusoidal positions, so a board atom means the same thing in a prefix as in a BOARD response. Value bins freeze an equiprobable partition of teacher WDL-derived scores in [-1, 1] with dedicated bins for the -1 and +1 atoms; a bin token decodes to its frozen mass-weighted mean, and a mate token decodes to ±1 regardless of distance. The tokenizer version pins the serialized board format; extending the id space does not change it. Board policies embed the prefix with a board-only table carrying a dedicated padding row, while their shared output head spans the whole vocabulary: actions are read from the move block and values from the value and mate blocks.

## Sequence traces

A sequence trace is a causal program over one root position, grouped into rounds. `<depth>` opens a round at the root; a move token enters the child reached by that legal move; `BACK` returns to the parent and is illegal at the root; `RESET` returns to the root from any depth; value and mate tokens evaluate the current node from the root player's perspective; and `<refuted>` states that a just-entered candidate did not beat the best exact candidates of its parent in the current visit. `BOARD` requests the current node's full board encoding; the root never requests one because its board is the prefix. Black-to-move roots store moves and observations rank-mirrored, while a board observation's side and counter atoms keep their absolute values.

A node visit runs from the move that enters the node to the BACK or RESET that leaves it. It records one of three outcomes for each child: an exact evaluation, a refutation, or a pass. Writing an evaluation or a refutation ends the visit; an evaluation is the visit's only one. A round's root evaluation is its last operation, after which only a new `<depth>` or `FINAL` is legal. Checkmate, stalemate, path repetition, and the 50-move rule are rule terminals: the searchable path stops there, and insufficient material is not a terminal.

`FINAL` is the decision point, not the end of the trace. It may follow any operation boundary and implicitly returns to the root. It is followed by a root move, a value token, an optional mate-distance token, the best line, and `<end>`. A root that can claim a draw writes the draw claim in place of the move and has no line: `FINAL <claim_draw> <value> [<mate>] <end>`. The suffix is a prediction region, so a truncated trace carries the final answer of its complete search regardless of where its thought was cut. `pondera.data.trace.TraceWriter` produces traces and `replay` rebuilds and validates them with python-chess from the root FEN alone. The writer, replay, and the generation mask share one visit state machine.

Sequence datasets store `board_tokens`, `trace_tokens`, `trace_len`, and `trace_roles` arrays with manifest `format_version = 4` and `grammar_version = 2`. Roles mark thought moves, evaluations, controls, refutations, the FINAL decision, the label evaluation, and the best line. Every trace ends with `<end>`, and the FINAL decision is not the last token.

Offline collection and synthesis are driven by `configs/trace_v2.toml`. The collector samples roots from an existing root release, runs one fresh instrumented engine process per root, and keeps the sidecar and root metadata in a run directory. The release builder turns those records into presentation rows — four per root without a recipe, or the single presentation selected by the recipe — and publishes independently hashed shards, resuming existing shards by root. Both commands are library entry points:

```bash
uv run python -m pondera.data.trace_collect --config configs/trace_v2.toml --roots ROOTS.jsonl \
    --instrumentation INSTRUMENTATION.patch --control CONTROL_BINARY
uv run python -m pondera.data.sequence_release --config configs/trace_v2.toml \
    --collection RUN/collection.json
```

Validate a completed release with `uv run python -m pondera.data.sequence_validation --release RELEASE --roots ROOTS.jsonl --workers 8`. This checks file hashes and array shapes, then replays every trace with python-chess to verify board observations, token roles, visit and refutation rules, the FINAL suffix, tool/no-tool equivalence, truncation boundaries, and the stored truncation states. `--roots`, `--start`, and `--end` are optional; without them, validation checks internal completeness rather than equality with an external root selection. When external roots are supplied, validation also checks their content hash where available and every stored provenance field, including the train/validation split. Training reads only the four board/trace arrays directly; the remaining arrays describe the search state at each truncation boundary, and each shard's `champions.jsonl` records the per-round champion audit used to validate those states.

## Train and resume

```bash
uv run python -m pondera.training.depth --config CONFIG.toml --data-dir DATA --output-dir RUN --device cuda
uv run python -m pondera.training.sequence --config CONFIG.toml --data-dir DATA --output-dir RUN --device cuda
uv run python -m pondera.training.sequence --config CONFIG.toml --data-dir DATA --output-dir RUN --device cuda --resume RUN/final.pth
```

`depth` trains board policies from position arrays; `sequence` trains causal trace policies from a release. Choose `--device cpu` for CPU execution. `--track` enables SwanLab; local metrics are always written. `--max-steps` stops at a positive absolute optimizer step without changing the configured schedule: resuming from step 100 with `--max-steps 110` runs 10 more steps, and a limit already reached runs none. Resume requires the same training configuration and input arrays. Checkpoints store optimizer, scheduler, progress, and random-number state. Partial accumulation groups at an epoch boundary are discarded.

Sequence training applies shifted next-token cross-entropy over generated trace tokens; the current mask excludes padding and tool responses and weights all other roles equally. Masking is a training choice, not part of the data format.

Microbatch size and gradient accumulation may be regrouped on resume when their product stays unchanged. The trainer converts the saved position cursor to the new microbatch size and preserves example order, optimizer updates, and the learning-rate schedule. Other configuration changes remain rejected. `execution.json` and checkpoint `execution_history` record regrouping. Floating-point reductions and dropout execution can differ after regrouping, so bitwise equivalence is not guaranteed.

Checkpoints use `format_version = 1` with `tokenizer_version`, `model_config`, and `model_state_dict`, plus training state when applicable. Model, dataset, and checkpoint tokenizer versions must agree. Model configuration selects `architecture = "depth"` or `"sequence"`.

For policy comparisons, `train_flops_budget` stops training after the first complete optimizer step reaching the specified analytical budget. The warmup and cosine schedule use consumed FLOPs rather than step count. Choose enough `epochs` to reach the budget; `--max-steps` still permits an intentional interruption. Metrics and checkpoints record consumed training FLOPs and examples, and resume checks dataset identity. Sequence training verifies the current checksums of every training-input file. The convention counts dense matrix multiplies, with multiply-add equal to two FLOPs and logical token lengths excluding padding; scalar operations and optimizer updates are excluded. Validation is outside the training budget. Runtime and memory are diagnostic measurements, not alignment criteria. See [token backbones](backbones.md) for architecture and execution contracts.

## Evaluate

UCI serving exposes a checkpoint as an engine:

```bash
uv run python -m pondera.inference.uci --checkpoint RUN/final.pth --device cuda
```

UCI inference supports depth-policy checkpoints only; sequence checkpoints are rejected rather than treated as direct-policy models. It takes the largest policy logit among legal moves, with no search, sampling, or pondering. Model configuration determines the depth and repetition limits; `go` time controls do not currently alter those limits.

Matches are orchestrated by the library function `pondera.evaluation.match.run_match`, which invokes the UCI adapter as a subprocess and drives fastchess. Matches use paired colors, sequential openings, one Stockfish thread, and a fixed opponent depth. The bundled opening suite is a small baseline fixture; pass a larger suite explicitly for serious comparisons. The engine version should be held fixed when comparing results. Match artifacts include the executed command, opening-suite hash, full engine output, PGN, and structured results. Failed or incomplete matches do not produce a successful result. Review results before adding them to the public leaderboard; search-free and search-enabled evaluations must be reported separately.
