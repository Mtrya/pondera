# Pondera

Pondera studies search-free chess policies and whether a neural model can internalize planning. Playing strength is a research measurement. The policy executes no explicit search during training or deployment; offline engine supervision and engine feedback for reinforcement learning are allowed.

The maintained implementation includes lossless square and compact-piece board inputs; base, sequence, recurrent-depth, and combined token backbones; offline Stockfish annotation with frozen teacher releases; supervised policy training with analytical FLOPs accounting; a deterministic UCI adapter; and fastchess match evaluation. ChessFormer-era checkpoints remain published on Hugging Face as provenance for the historical baselines recorded in the [leaderboard](results/leaderboard.md); the current codebase implements only the versioned lossless board format.

## Setup

```bash
git clone https://github.com/Mtrya/pondera.git
cd pondera
uv sync --all-extras --group dev
```

Python 3.13 or later is required. Stockfish and fastchess are external executables; install them separately and put them on `PATH`. A minimal inference installation needs only `uv sync --no-dev`; data processing and training dependencies are available through the `data` and `training` extras.

## Usage

The package is a library. Training runs from a TOML configuration:

```bash
uv run python -m pondera.training.supervised --config CONFIG.toml --data-dir DATA --output-dir RUN --device cuda
```

Experiment configurations live with their run directories; the repository tracks configurations only for published reference runs. See [usage](docs/usage.md) for data formats, teacher releases, and evaluation semantics, and [architecture](docs/architecture.md) for module boundaries and repository conventions.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run lint-imports
uv run python -m unittest discover -s tests -v
```

Tests use real models, local datasets, and subprocesses. Stockfish and fastchess integration tests run when their executables are on `PATH`. Public documentation contains stable usage and design information. Session notes, machine-specific instructions, and intermediate work stay untracked.
