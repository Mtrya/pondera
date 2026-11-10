# Repository architecture

Pondera is an installable Python package rooted at `src/pondera`. It is a library: training, trace data preparation, and UCI serving expose `python -m` entry points, and experiment orchestration lives in run directories outside the package. Imports do not start engines, download checkpoints, or create outputs.

| Location | Responsibility |
|---|---|
| `chess/` | Action vocabulary, rule-only path tracking, lossless board token encoding, exact symmetry transforms, and min-cost merge dictionaries |
| `models/` | Neural model definitions |
| `data/` | Root preparation, trace curation, and vocabulary fitting plus encoding under one TOML recipe; release validation, authoritative text conversion, and packed training datasets |
| `tools/export_hf.py` | Bounded projection of an accepted local release into the shared eleven-column Hugging Face dataset; schema definitions remain in `data/` |
| `training/` | Supervised optimization and resumable training state |
| `inference/` | Checkpoint loading, legal move selection, and the UCI protocol |
| `evaluation/` | External match orchestration, graded puzzles, Syzygy adjudication, and opening assets |
| `tests/` | Behavioral and integration tests |
| `results/` | Reviewed summaries of completed evaluations |

Representation code does not import application layers, and models do not import workflows. Training and inference use the model definitions. Training reads packed datasets and dataset identity checks from `data`; it does not call collection, synthesis, search, or evaluation code. Evaluation orchestrates inference and external engines. Data and evaluation do not import training. Import contracts enforce these boundaries.

The board tokenizer has one format shared by offline data preparation and runtime inputs: sparse self-contained `(square, piece)` atoms plus metadata atoms, preserving piece placement and the first four FEN fields exactly. Base token IDs and names are generated directly from the arithmetic layout in `chess/vocabulary.py`. The fullmove counter is not encoded; halfmove clocks of 100 and above and repetition counts above four share their respective top atoms.

Both policy baselines are fixed-depth causal Transformers over one packed token stream, with NoPE, pre-norm RMSNorm, SiTU-GLU feed-forward networks, and grouped-query attention. The vanilla model uses causal softmax attention. The linear-attention model uses an identity feature map, no decay and unnormalized readout, with fp32 chunk and recurrent states. Model configuration contains dimensions and vocabulary size; the checkpoint's `arch` field selects the implementation.

Training prepares sequence metadata on the host once per forward. Softmax attention reuses the maximum document length in every layer. Linear attention batches independent documents in groups rounded to powers of two chunks, keeping each document's state separate; padding is internal to the attention computation. Distributed updates plan global batch indices from length metadata, read only each rank's tokens, and reduce accumulated gradients once per optimizer update.

Packed-trainer checkpoints use `format_version = 2` and retain model and optimizer states, the actual number of presented traces, dataset identity, tokenizer versions and merge entries, selected document indices, training configuration, and the SwanLab run ID. Resume requires the same architecture, model dimensions, dataset, tokenizer, selected documents, sampling seed, and microbatch token budget. Requested learning rates and weight decay override the saved optimizer settings. Local `config.json` and SwanLab share the same configuration, including architecture; `history.jsonl` is the local training history, and resume discards records beyond the loaded checkpoint before appending.

Generation reuses the board encoding, appends a standalone `FINAL`, masks the next-token distribution to legal decisions, and returns the highest-scoring move. Black move scores use the reflected move IDs used in training; the board prefix and returned UCI move remain in absolute coordinates. The UCI protocol loop is independent of checkpoint loading and move selection; `pondera.inference.vanilla_uci` serves these checkpoints, preserves repetition counts from the supplied move history, returns `0000` for finished games, and is the default match engine.

## Repository hygiene

Maintain reusable code inside the package. Do not add root Python scripts or generic utility packages. Temporary scripts, downloaded data, checkpoints, logs, external tools, and research working notes use `.local/scratch`, `.local/data`, `.local/runs`, `.local/research`, and `.local/tools`, or explicitly selected external storage. Run directories contain their configuration and outputs; evaluation does not automatically modify reviewed result summaries. Experiment configurations live with their run directories; the repository tracks configurations only for published reference runs.

Research records have two layers. Working notes — the experiment ledger, open research threads, and session handoff — live untracked under `.local/research/`. Converged, publicly readable conclusions are promoted to `docs/research/`. Everything else in `docs/` is stable, self-contained, and appropriate for public readers. Ignore rules use explicit exclusions. Being visible to Git is not approval to publish a document. The same public-content standard applies to code, comments, configurations, tests, and pull requests: no references to session-specific plans or unavailable internal material.

Retire unused implementations after checking their callers and required baseline behavior. Git history retains previous implementations. Add tests for observable behavior and data contracts, using actual implementations and local fixtures; do not encode prose or incidental directory structure as tests.
