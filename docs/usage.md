# Usage and data formats

Run commands from an activated environment or prefix them with `uv run`. Paths are supplied explicitly or default to `.local/` beneath the working directory. Engine executables are resolved from `PATH` unless a path is supplied. The examples assume the data and training extras are installed.

## Prepare position data

Human extraction and best-move annotation produce Hugging Face datasets with `fen`, `next_move`, and `count` columns. They use stable hash partitions and resumable manifests. `pondera-human --help` and `pondera-stockfish --help` describe source selection and output options. Publishing is explicit through `pondera-publish` or the processors' `--push-to-hub` option.

Prepare policy training arrays from local parquet files:

```bash
pondera-prepare --train-file /datasets/train.parquet --val-file /datasets/validation.parquet --output-dir .local/data/policy
```

Repeated file arguments concatenate shards in the given order. `--train-rows` selects positions at evenly spaced indices over the complete training source. Unknown target moves fail preparation. Outputs are `tokens_train.npy`, `moves_train.npy`, `tokens_val.npy`, `moves_val.npy`, and a manifest. Use an empty output directory.

Alternatively, download a Hub dataset:

```bash
pondera-prepare --dataset kaupane/chess-positions --revision main --train-rows 15000000 --output-dir .local/data/policy
```

For reproducible data acquisition, replace `main` with a repository commit. The default splits are `stockfish_train` and `stockfish_val`; downloaded shards use the selected cache directory.

Token arrays have shape `(positions, 73)` and dtype `uint8`. They encode 64 squares, side to move, four castling rights, en passant, halfmove clock, fullmove number, and repetition. Move arrays contain `int16` indices into the 1,969-action vocabulary, including the historical draw-claim action. UCI emits board moves only and leaves draw adjudication to the match runner.

## Annotate offline positions

```bash
pondera-annotate /datasets/positions.parquet --engine stockfish --depth 16 --multipv 5 --workers 4 --output .local/data/annotations/labels.jsonl
```

The source needs `fen`, `next_move`, and `count`. Duplicate FENs are annotated once. Records contain source metadata and a `unit` with `per_depth`, `reached_depth`, and `nodes`. Each depth has `r1` through the available candidate ranks, containing `bm`, `cp`, and `mate`; scores are from White's perspective, with only one of `cp` and `mate` populated. The deepest recorded candidates also contain their principal variation in `pv`. Terminal positions are marked separately. A time limit can truncate the requested depth, so consumers must inspect `reached_depth` and available ranks.

A sidecar manifest identifies the teacher and its configuration. Resuming requires the same configuration and skips recorded FENs. Engine and malformed-input failures propagate instead of silently producing labels. Annotation is a separate offline command and is not called by the training loop.

## Train and resume

```bash
pondera-train --config configs/supervised.toml --data-dir .local/data/policy --output-dir .local/runs/supervised --device cuda
pondera-train --config configs/supervised.toml --data-dir .local/data/policy --output-dir .local/runs/supervised --device cuda --resume .local/runs/supervised/final.pth
```

Choose `--device cpu` for CPU execution. `--track` enables SwanLab; local metrics are always written. `--max-steps` stops at an absolute optimizer step without changing the configured schedule. Resume requires the same training configuration and input arrays. Checkpoints store optimizer, scheduler, progress, and random-number state. Partial accumulation groups at an epoch boundary are discarded.

New checkpoints use `format_version = 1`, `model_config`, and `model_state_dict`, plus training state when applicable. The inference loader also accepts existing baseline checkpoint files and Hugging Face model directories or repository identifiers.

## Evaluate

```bash
pondera-uci --checkpoint kaupane/ChessFormer-SL --device cpu
pondera-evaluate /datasets/validation.parquet --checkpoint .local/runs/supervised/final.pth --device cuda
pondera-match --checkpoint .local/runs/supervised/final.pth --sf-depth 1 --games 100 --device cuda --output-dir .local/runs/match
```

UCI inference takes the largest policy logit among legal moves, with no search, sampling, or pondering. `go` time controls do not change its fixed forward-pass budget. Evaluation reports policy cross entropy, top-1 accuracy, and illegal-move probability, weighted by position count.

Matches use paired colors, sequential openings, one Stockfish thread, and a fixed opponent depth. The bundled opening suite is a small baseline fixture; pass `--openings` for a larger suite. The engine version should be held fixed when comparing results. Match artifacts include the executed command, opening-suite hash, full engine output, PGN, and structured results. Failed or incomplete matches do not produce a successful result. Review results before adding them to the public leaderboard; search-free and search-enabled evaluations must be reported separately.
