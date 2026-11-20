# Checkpoint evaluation

Compute full-document next-token loss with:

```bash
uv run python -m pondera.evaluation.loss --checkpoint RUN/checkpoint.pth --packed ENCODED --output LOSS --device cpu
```

`loss.py` evaluates every validation document with its complete causal context and the checkpoint's vocabulary. The reported loss is total next-token negative log likelihood divided by the number of labels, excluding only each document's terminal input position. Training selection limits and training-loss weights do not change this metric. `summary.json` records token counts, runtime, precision, and model settings; `documents.jsonl` retains each document's label count and summed negative log likelihood for paired comparisons. `--batch-tokens` controls packing, and a longer document receives a batch large enough to keep it intact. `--loop-repetitions` overrides inference depth for a looped checkpoint.

Launch with `torchrun --standalone --nproc-per-node=4 -m pondera.evaluation.loss ... --device npu` (or `cuda`) to divide complete packed batches across devices. Each process selects its local device; Gloo exchanges only per-document CPU results, and rank zero writes the combined output. Packing, document order, and token weighting match single-process evaluation. The summary records `world_size` and elapsed evaluation time. Programmatic callers can initialize a process group before calling `evaluate_loss` on every rank.

Evaluate playing behavior with:

```bash
uv run python -m pondera.evaluation --config configs/eval.toml --checkpoint RUN/checkpoint.pth --roots HELD_OUT_ROOTS --puzzles PUZZLES --cache REFERENCE.sqlite --output EVALUATION --engine STOCKFISH --device cpu
```

Provide roots, puzzles, or both. Inputs may be JSONL or Parquet; curated Parquet directories and prepared root pools are accepted for roots. Roots provide `fen` and optional `repetition` (default one). Puzzles use the Lichess `FEN`, `Moves`, `Rating`, and optional `PuzzleId` fields. The first stored puzzle move sets up the actual root. Use held-out inputs: the evaluator does not split a training release automatically. Selection is deterministic, bounded by `[evaluation].limit`, and deduplicates roots or puzzle IDs before evaluation. Terminal roots are rejected.

`configs/eval.toml` owns generation allowances, sample limits, engine resources, PV scoring depth, and puzzle rating buckets. Checkpoint/input/output/cache/engine paths and device selection remain command arguments. Every selected input is evaluated with zero reasoning and the configured reasoning budget, using the same unconstrained inference path.

The output contains `summary.json`, `positions.jsonl`, and, unless an external cache is supplied, `reference.sqlite`. Completed position records are written immediately, retaining model tokens, generation counters, parsed answers, failure codes, and scores. The summary is replaced atomically after each reasoning budget; its `status` is `running`, `complete`, or `failed`. Failed runs retain their completed position evidence and an error record. An output directory alone does not indicate completion.

Per-position `decision_quality` grades every legal selected move independently of trailing value, PV, or END errors. `quality` additionally grades the value and PV when the full answer is valid. Summary move metrics use all legal decisions, with `decision_quality_count` and `decision_failure_count` showing coverage over all selected initial positions. Value and PV metrics use complete answers, counted by `answer_quality_count`. Each metric reports its own measured count; unscorable decisions receive no invented engine score. Format validity, legal decisions, legal PVs, and valid complete answers also use all selected initial positions as their denominator. Puzzle correctness and progression depend on the legal selected move even when the answer suffix is malformed. Whole-puzzle accuracy queries each solver turn against the supplied opponent replies; it is separate from the PV emitted in a single answer. Any legal checkmating alternative is accepted.

`reference.py` evaluates each legal move with fresh Stockfish search state and the same node budget. Its `grade_decision(fen, repetition, decision)` accepts a legal UCI move or an available draw claim and rejects terminal roots or invalid decisions. Cached scores are reused across checkpoints and distinguish engine builds, search settings, positions, and supplied move history. Root move quality reports centipawn loss where both scores are finite, expected-score loss, and rank among legal decisions. Mate cases retain explicit mate scores. Value error compares the generated W−L value with the best searched root value, matching the corpus target; it is independent of the chosen move's quality. A legal draw claim participates in move-quality scoring with value zero, while the value target retains the searched root value, as in curation. Prior repetitions are not fabricated as an invented move history for Stockfish.

All PV continuations are replayed for legality. Engine quality is measured for at most `[engine].pv_plies` continuation plies, with losses reported separately for the root player and the opponent. Set that limit to zero to skip these searches. A favorable PV endpoint alone is not a quality score. The reference engine is a fixed judge; its expected scores are not calibrated winning probabilities for the model being evaluated.

`checkpoint.py` orchestrates selection, generation, and scoring; `config.py` parses the recipe; `__main__.py` exposes the entry point. `puzzles.py` also offers the move-policy puzzle API. `match.py` retains the fastchess match API, whose current default opponent plays against the legal-masked zero-presentation UCI policy. Checkpoint evaluation has no dependency on training or telemetry.
