# Chess representation

The representation layer. It depends only on `python-chess`, `numpy`, and the
standard library; it does not import data, model, training, inference, or
evaluation code.

| File | Responsibility |
|---|---|
| `domain.py` | Piece and square index tables used by the encoder and model. |
| `vocabulary.py` | Generate canonical base token names and move-ID maps from the arithmetic layout. `python -m pondera.chess.vocabulary` prints the names in ID order. |
| `codec.py` | Lossless board and metadata atoms, FEN/repetition decoding, legal move indices, and the numeric meaning of value and mate tokens. |
| `symmetry.py` | Black-to-white token mirrors, UCI move reflection, and the move permutation table. |
| `tracker.py` | `PathTracker`, the rule-only root-plus-path state machine: repetition readings, search terminals, legal moves, and board observations. `repetition_count` counts a full python-chess move history. |
| `merge.py` | The composite-token layer above the atom vocabulary: flat atom expansions, candidate counting, encoder-driven dictionary selection, minimum-token encoding, and decoding. Dataset metadata and checkpoints retain the expansions. |
