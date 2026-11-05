# Lichess puzzle fixtures

Twelve genuine rows from [Lichess/chess-puzzles](https://huggingface.co/datasets/Lichess/chess-puzzles), revision `461725966ea46898ec6481372a6326de3281bfbd`, retrieved through the Dataset Viewer with `config=default`, `split=train`, `offset=0`, `length=12`. The source is CC0.

The stored FEN precedes the opponent move; applying the first UCI move gives the position to solve. The fixtures retain every source field, including nullable opening tags. Tests use bounded rows to qualify preparation, provenance, deterministic selection, and the curation/encoding path. A constructed halfmove-clock edge case is derived in its test rather than stored as an original source row.
