# Pondera

Pondera studies search-free chess policies and whether a neural model can internalize planning. Playing strength is a research measurement. The policy executes no explicit search during training or deployment; offline engine supervision and engine feedback for reinforcement learning are allowed.

**Status.** Pondera is a research work in progress. The package is experiment infrastructure, not a usable product: there is no stable user-facing API or command set, the data formats and model interfaces still change, and nothing here promises backward compatibility. The code is published so experiments can be inspected and reproduced, not as a turnkey chess engine.

The current implementation includes lossless sparse board atoms, causal softmax and linear-attention Transformer baselines over one packed token stream, a trace grammar with a shared visit state machine and replay validator, supervised training with resumable checkpoints, a deterministic UCI adapter, graded puzzles, and fastchess match evaluation. ChessFormer-era checkpoints remain published on Hugging Face as provenance for the historical baselines recorded in the [leaderboard](results/leaderboard.md); the current codebase implements only the versioned lossless board format.

## Setup

```bash
git clone https://github.com/Mtrya/pondera.git
cd pondera
uv sync --group dev
```

Python 3.13 or later is required. Stockfish and fastchess are external executables; install them separately and put them on `PATH`. A minimal inference installation needs only `uv sync --no-dev`.

## Usage

The package is a library. Train a model from a packed dataset:

```bash
uv run python -m pondera.training.packed --config configs/train.toml --packed ENCODED --output RUN --device cuda
```

Model architecture and dimensions, data selection, optimization, token budgets, stopping targets, validation cadence, and checkpoint retention are configured in [configs/train.toml](configs/train.toml). Dataset/output/checkpoint paths, device selection, and `--track` / `--no-track` remain command arguments. Resume uses the same recipe with `--resume RUN/checkpoint.pth`; change tunables in the TOML, and incompatible model or sampling settings are rejected.

Muon and AdamW use linear warmup followed by cosine decay. Their configured learning rates are peaks; `warmup_fraction` defaults to 0.05 and `min_lr_ratio` to 0.1. The schedule follows the configured update or presentation budget, and actual rates are logged with each optimizer update.

Shared data and training recipes live in `configs/`. Experiment configurations live with their run directories; only published reference runs add experiment configurations to the repository. See [data synthesis](docs/data.md) for the current offline pipeline and [architecture](docs/architecture.md) for module boundaries and repository conventions.

Evaluate a checkpoint's unconstrained answers on held-out roots and/or Lichess puzzles:

```bash
uv run python -m pondera.evaluation --config configs/eval.toml --checkpoint RUN/checkpoint.pth --roots HELD_OUT_ROOTS --cache REFERENCE.sqlite --output EVALUATION --engine STOCKFISH --device cpu
```

The evaluator compares zero reasoning with the configured thinking budget, reports answer validity, engine-scored move quality, root-value accuracy, and PV quality, and reuses engine references across checkpoints. Add `--puzzles PUZZLES` for first-move and whole-puzzle success. See [evaluation](src/pondera/evaluation/README.md) for input and scoring contracts.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run lint-imports
uv run python -m unittest discover -s tests -v
```

Tests use real models, local datasets, and subprocesses. Stockfish and fastchess integration tests run when their executables are on `PATH`. Public documentation describes the current design and evolves with the code. Session notes, machine-specific instructions, and intermediate work stay untracked.
