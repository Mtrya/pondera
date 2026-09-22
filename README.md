# Pondera

Pondera studies search-free chess training and whether a neural model can internalize planning. Playing strength is a research measurement. Training uses no search-based policy improvement; optional inference-time search must be reported separately from search-free inference.

The maintained implementation includes a transformer policy/value baseline, offline position preparation and Stockfish annotation, supervised policy training, a deterministic UCI adapter, and fastchess evaluation. Existing ChessFormer checkpoints remain readable as historical baselines. The current policy trainer optimizes move cross entropy only.

## Setup

```bash
git clone https://github.com/Mtrya/pondera.git
cd pondera
uv sync --all-extras --group dev
```

Python 3.13 or later is required. Stockfish and fastchess are external executables; install them separately and put them on `PATH`, or pass their paths to the corresponding command. A minimal inference installation needs only `uv sync --no-dev`; data processing and training dependencies are available through the `data` and `training` extras.

## Commands

| Command | Purpose |
|---|---|
| `pondera-human` | Extract and aggregate human moves |
| `pondera-stockfish` | Produce best-move labels from a position dataset |
| `pondera-publish` | Explicitly upload prepared datasets to Hugging Face |
| `pondera-prepare` | Encode parquet positions into memory-mapped training arrays |
| `pondera-annotate` | Record Stockfish scores across depths and final principal variations |
| `pondera-train` | Train or resume the supervised policy baseline |
| `pondera-uci` | Serve deterministic legal policy moves over UCI |
| `pondera-evaluate` | Measure policy loss, accuracy, and illegal-move probability |
| `pondera-match` | Run paired matches and save results and game records |

Use `uv run <command> --help` for arguments. Training settings live in [configs/supervised.toml](configs/supervised.toml). See [usage](docs/usage.md) for complete examples and data formats, and [architecture](docs/architecture.md) for module boundaries and repository conventions.

## Historical baselines

The existing model has approximately 100.7 million parameters, 20 transformer blocks, 640 hidden dimensions, and a 1,969-action policy head. Its input contains 73 position tokens and two learned readout tokens. Published checkpoints are [ChessFormer-SL](https://huggingface.co/kaupane/ChessFormer-SL) and [ChessFormer-RL](https://huggingface.co/kaupane/ChessFormer-RL).

Measured match results are recorded in the [leaderboard](results/leaderboard.md). These are small-sample comparisons against specified Stockfish depths, not absolute Elo ratings.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run lint-imports
uv run python -m unittest discover -s tests -v
```

Tests use real models, local datasets, and subprocesses. Stockfish and fastchess integration tests run when their executables are on `PATH`. Public documentation contains stable usage and design information. Session notes, machine-specific instructions, and intermediate work stay untracked.
