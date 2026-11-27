# Chess representation

The representation layer owns the standard chess rules, the board and move token encodings, and path tracking. The Python package imports the Rust extension as `pondera.chess._rules`.

| File | Responsibility |
|---|---|
| [Rust chess](../../../rust/chess/) | Standard positions, legal moves, SAN, FEN, game history, and whole-game PGN replay. PyO3 exposes `Board`, `Move`, and `replay_mainline`. |
| `domain.py` | Piece and square index tables used by the encoder and model. |
| `vocabulary.py` | Generate canonical base token names and move-ID maps from the arithmetic layout. `python -m pondera.chess.vocabulary` prints the names in ID order. |
| `codec.py` | Lossless board and metadata atoms, FEN/repetition decoding, legal move indices, and the numeric meaning of value and mate tokens. |
| `symmetry.py` | Black-to-white token mirrors, UCI move reflection, and the move permutation table. |
| `tracker.py` | `PathTracker`, the rule-only root-plus-path state machine: repetition readings, search terminals, legal moves, and board observations. `repetition_count` counts occurrences in the board's move history. |
| `merge.py` | The composite-token layer above the atom vocabulary: flat atom expansions, candidate counting, encoder-driven dictionary selection, minimum-token encoding, and decoding. Dataset metadata and checkpoints retain the expansions. |
