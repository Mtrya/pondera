# Repository architecture

Pondera is an installable Python package rooted at `src/pondera`. Command entry points are registered in `pyproject.toml`; commands parse arguments and call domain implementations. Imports do not start engines, download checkpoints, or create outputs.

| Location | Responsibility |
|---|---|
| `chess/` | Action vocabulary, position history, and FEN token encoding |
| `models/` | Neural model and checkpoint serialization |
| `data/` | Dataset preparation, offline annotation, storage, and publishing |
| `training/` | Supervised optimization and resumable training state |
| `inference/` | Legal move selection and UCI protocol |
| `evaluation/` | Policy metrics, external match orchestration, and opening assets |
| `cli/` | Command-line interfaces |
| `configs/` | Reproducible public experiment settings |
| `tests/` | Behavioral and integration tests |
| `results/` | Reviewed summaries of completed evaluations |

Representation code does not import application layers. Models depend on representation, while training and inference depend on models. Training does not call search, annotation, or evaluation code. Evaluation orchestrates inference and external engines. Import contracts enforce these boundaries.

FEN parsing has one implementation. Model embeddings consume its integer representation, whether callers supply FEN strings or prepared token arrays. A single checkpoint loader serves inference and evaluation; the writer is shared with training. Existing checkpoint configuration keys are accepted to preserve historical baseline evaluation, while new checkpoints use a single versioned format.

## Repository hygiene

Maintain reusable code inside the package. Do not add root Python scripts or generic utility packages. Temporary scripts, downloaded data, checkpoints, logs, and external tools use `.local/scratch`, `.local/data`, `.local/runs`, and `.local/tools`, or explicitly selected external storage. Run directories contain their configuration and outputs; evaluation does not automatically modify reviewed result summaries.

Ignore rules use explicit exclusions. Documentation suitable for version control must be stable, self-contained, and appropriate for public readers. Private working documents are excluded by `docs/.gitignore`. Being visible to Git is not approval to publish a document. The same public-content standard applies to code, comments, configurations, tests, and pull requests: no references to session-specific plans or unavailable internal material.

Retire unused implementations after checking their callers and required baseline behavior. Git history retains previous implementations. Add tests for observable behavior and data contracts, using actual implementations and local fixtures; do not encode prose or incidental directory structure as tests.
