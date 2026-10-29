# Usage and data formats

Pondera is a library. Training and UCI serving have `python -m` entry points; annotation, teacher-release preparation, and match evaluation are library functions that experiment run scripts compose. Paths are supplied explicitly or default to `.local/` beneath the working directory. Engine executables are resolved from `PATH` unless a path is supplied. The examples assume the training extra is installed.

## Annotate positions

`pondera.data.annotation.annotate_positions(records, output, config, workers)` annotates positions with a fixed Stockfish configuration (`AnnotationConfig`: engine, depth, multipv, threads, hash, time limit). Records need `fen`, `next_move`, and `count`. Duplicate FENs are annotated once. Output records contain source metadata and a `unit` with `per_depth`, `reached_depth`, and `nodes`. Each depth has `r1` through the available candidate ranks, containing `bm`, `cp`, and `mate`; scores are from White's perspective, with only one of `cp` and `mate` populated. The deepest recorded candidates also contain their principal variation in `pv`. Terminal positions are marked separately. A time limit can truncate the requested depth, so consumers must inspect `reached_depth` and available ranks.

A sidecar manifest identifies the teacher and its configuration. Resuming requires the same configuration and skips recorded FENs. An interrupted final record without a newline is discarded and annotated again; malformed complete records raise an error. Engine and malformed-input failures propagate instead of silently producing labels. Annotation is offline and is never called by the training loop.

## Teacher releases

A frozen teacher release is a manifest plus annotated shards. `pondera.data.teacher.prepare_teacher_depth(release_path, output_dir, max_depth=16, board_aux=False)` verifies source checksums, target legality, and absence of overlap by the first four FEN fields, then writes lossless token arrays, per-depth rank-one move indices, value targets as both vocabulary tokens and scalars, legal-action masks, and a provenance manifest. With `board_aux`, each row additionally carries the square contents after replaying each depth's teacher move plus the changed-square mask. Positions without game history use repetition count 1; that is an explicit input limitation, not recovered history. Use an empty output directory. A release is frozen: newly collected shards form later releases and never silently join an in-progress comparison.

The board tokenizer produces `int16` arrays padded to the longest encoded position in each split. Each row is the set of occupied squares as self-contained atoms, `square * 12 + piece` in a1..h8 scan order, followed by eight metadata atoms: side to move, four castling rights, en passant, halfmove clock, and repetition count. Empty squares carry no token. The encoding is lossless for the first five FEN fields; the fullmove counter is not encoded, so decoding restores the FEN modulo that counter, and repetition counts above four share the top atom. The input contract is FEN plus repetition count, not full game history. Padding is masked.

Move arrays contain `int16` indices into the 1,969-action vocabulary, including the historical draw-claim action. UCI emits board moves only and leaves draw adjudication to the match runner. Manifests include the tokenizer version and array checksums.

Beyond board encodings, `pondera.chess.vocabulary` freezes one shared id space (CHESS_TOKENIZER) for sequence models: 949 board and metadata atoms, 1,969 move tokens, 20 value-bin tokens, mate-distance tokens (win 0–28, loss 0–30; distance 0 is reserved), and four trace delimiters (PAD, END, LINE, FINAL), for 3,002 ids in total. Sequence models embed the whole space with one table; the board prefix carries no positional term, while trace tokens add sinusoidal positions, so a board atom means the same thing in a prefix as in a snapshot. Value bins freeze an equiprobable partition of teacher WDL-derived scores in [-1, 1] with dedicated bins for the -1 and +1 atoms; a bin token decodes to its frozen mass-weighted mean, and a mate token decodes to ±1 regardless of distance. The tokenizer version pins the serialized board format; extending the id space does not change it. Board policies embed the prefix with a board-only table carrying a dedicated padding row, while their shared output head spans the whole vocabulary: actions are read from the move block and values from the value and mate blocks.

## Train and resume

```bash
uv run python -m pondera.training.depth --config CONFIG.toml --data-dir DATA --output-dir RUN --device cuda
uv run python -m pondera.training.depth --config CONFIG.toml --data-dir DATA --output-dir RUN --device cuda --resume RUN/final.pth
```

Choose `--device cpu` for CPU execution. `--track` enables SwanLab; local metrics are always written. `--max-steps` stops at a positive absolute optimizer step without changing the configured schedule: resuming from step 100 with `--max-steps 110` runs 10 more steps, and a limit already reached runs none. Resume requires the same training configuration and input arrays. Checkpoints store optimizer, scheduler, progress, and random-number state. Partial accumulation groups at an epoch boundary are discarded.

Microbatch size and gradient accumulation may be regrouped on resume when their product stays unchanged. The trainer converts the saved position cursor to the new microbatch size and preserves example order, optimizer updates, and the learning-rate schedule. Other configuration changes remain rejected. `execution.json` and checkpoint `execution_history` record regrouping. Floating-point reductions and dropout execution can differ after regrouping, so bitwise equivalence is not guaranteed.

Checkpoints use `format_version = 1` with `tokenizer_version`, `model_config`, and `model_state_dict`, plus training state when applicable. Model, dataset, and checkpoint tokenizer versions must agree. Model configuration selects `architecture = "depth"` or `"sequence"`.

For policy comparisons, `train_flops_budget` stops training after the first complete optimizer step reaching the specified analytical budget. The warmup and cosine schedule use consumed FLOPs rather than step count. Choose enough `epochs` to reach the budget; `--max-steps` still permits an intentional interruption. Metrics and checkpoints record consumed training FLOPs and examples, and resume checks the dataset manifest identity. The convention counts dense matrix multiplies, with multiply-add equal to two FLOPs and logical token lengths excluding padding; scalar operations and optimizer updates are excluded. Validation is outside the training budget. Runtime and memory are diagnostic measurements, not alignment criteria. See [token backbones](backbones.md) for architecture and execution contracts.

## Evaluate

UCI serving exposes a checkpoint as an engine:

```bash
uv run python -m pondera.inference.uci --checkpoint RUN/final.pth --device cuda
```

UCI inference takes the largest policy logit among legal moves, with no search, sampling, or pondering. Model configuration determines its depth and repetition limits; `go` time controls do not currently alter those limits.

Matches are orchestrated by the library function `pondera.evaluation.match.run_match`, which invokes the UCI adapter as a subprocess and drives fastchess. Matches use paired colors, sequential openings, one Stockfish thread, and a fixed opponent depth. The bundled opening suite is a small baseline fixture; pass a larger suite explicitly for serious comparisons. The engine version should be held fixed when comparing results. Match artifacts include the executed command, opening-suite hash, full engine output, PGN, and structured results. Failed or incomplete matches do not produce a successful result. Review results before adding them to the public leaderboard; search-free and search-enabled evaluations must be reported separately.
