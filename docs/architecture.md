# Repository architecture

Pondera is an installable Python package rooted at `src/pondera`. It is a library: training, trace data preparation, and UCI serving expose `python -m` entry points, and experiment orchestration lives in run directories outside the package. Imports do not start engines, download checkpoints, or create outputs.

| Location | Responsibility |
|---|---|
| `chess/` | Action vocabulary, rule-only path tracking, lossless board token encoding, exact symmetry transforms, and min-cost merge dictionaries |
| `models/` | Neural model and checkpoint serialization |
| `data/` | Offline trace grammar, release validation, encoded training arrays, merge-vocabulary profiling, and root corpus construction and stratified pool sampling from game archives |
| `training/` | Supervised optimization and resumable training state |
| `inference/` | Legal move selection, masked trace generation and budget assessment, and the UCI protocol |
| `evaluation/` | External match orchestration and opening assets |
| `tests/` | Behavioral and integration tests |
| `results/` | Reviewed summaries of completed evaluations |

Representation code does not import application layers. Models depend on representation, while training and inference depend on models. Training reads encoded datasets and dataset identity checks from `data`; it does not call collection, synthesis, search, or evaluation code. Evaluation orchestrates inference and external engines. Data and evaluation do not import training. Import contracts enforce these boundaries.

The board tokenizer has one format shared by offline data preparation and runtime inputs: sparse self-contained `(square, piece)` atoms plus metadata atoms, preserving piece placement and the first four FEN fields exactly. The fullmove counter is not encoded; halfmove clocks of 100 and above and repetition counts above four share their respective top atoms. A single checkpoint format (`format_version = 1`) serves training, inference, and evaluation; older formats are not read.

Token backbones compose attention blocks, repeated cores, and explicit KV access policies. Board policies attach lossless board embeddings and action/value readouts. Sequence backbones accept embeddings without prescribing thought-token semantics or training-only heads. Their contracts live in the module docstrings.

## Repository hygiene

Maintain reusable code inside the package. Do not add root Python scripts or generic utility packages. Temporary scripts, downloaded data, checkpoints, logs, external tools, and research working notes use `.local/scratch`, `.local/data`, `.local/runs`, `.local/research`, and `.local/tools`, or explicitly selected external storage. Run directories contain their configuration and outputs; evaluation does not automatically modify reviewed result summaries. Experiment configurations live with their run directories; the repository tracks configurations only for published reference runs.

Research records have two layers. Working notes — the experiment ledger, open research threads, and session handoff — live untracked under `.local/research/`. Converged, publicly readable conclusions are promoted to `docs/research/`. Everything else in `docs/` is stable, self-contained, and appropriate for public readers. Ignore rules use explicit exclusions. Being visible to Git is not approval to publish a document. The same public-content standard applies to code, comments, configurations, tests, and pull requests: no references to session-specific plans or unavailable internal material.

Retire unused implementations after checking their callers and required baseline behavior. Git history retains previous implementations. Add tests for observable behavior and data contracts, using actual implementations and local fixtures; do not encode prose or incidental directory structure as tests.
