# Pondera

A research project on **search-free chess training**: can a neural network learn to play strong chess without ever running a search algorithm during training — and can "search" itself be internalized into the model's forward computation?

The project combines supervised distillation from Stockfish with reinforcement learning from self-play (no MCTS in the training loop), and studies architectures that allocate computation adaptively (pondering). Inference-time search is treated as an optional, measurable configuration rather than a training component.

## Status

**Rebootstrapping.** The v1 model (formerly "ChessFormer", ~100M params, SL-distilled + PPO) established a codebase and initial results, but its evaluation pipeline was broken and its RL stage degraded from the SL checkpoint. Current work:

- Rigorous evaluation infrastructure (fastchess-based Elo ladders vs Stockfish 17.1)
- A small SL probe to calibrate distilled-policy strength against data scale
- Literature-driven architecture research (recurrent pondering, auxiliary targets, value distributions)

See `notes/` for the working documents (handoff, provisioning, literature survey).

## v1 Architecture (legacy)

- **Size**: 100.7M parameters (20 blocks, 640 hidden, 8 heads, SwiGLU FFN 1728)
- **Input**: FEN tokenized into 73 tokens (64 squares + side/castling/en-passant/clocks/repetition) + 2 learned readout tokens
- **Output**: policy head over 1,969 structurally valid moves + scalar value head
- **Key constraint**: no search during training; inference used argmax policy or a shallow beam

## Installation & Setup

```bash
git clone https://github.com/Mtrya/chess-transformer
cd chess-transformer
uv sync
```

To run the interactive demo:

```bash
uv run python app.py
```

## Usage

```python
import torch
from model import PonderaModel

model = PonderaModel.from_pretrained("kaupane/ChessFormer-SL")  # legacy v1 checkpoint
model.eval()

fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
repetitions = torch.tensor([1])
with torch.no_grad():
    move_logits, position_value = model(fens, repetitions)
```

## Legacy v1 results (historical, informal)

These numbers predate the evaluation infrastructure and should be treated as indicative only:

| Model | Action Loss | Value Loss | Invalid Loss |
|-------|-------------|------------|--------------|
| ChessFormer-SL (v1) | 1.6985 | 0.0407 | 0.0303 |
| ChessFormer-RL (v1, from SL checkpoint) | 1.8329 | 0.0501 | 0.0484 |

- **ChessFormer-SL**: reasonable opening/endgame play, frequent midgame tactical blunders; strength was informally estimated around 1500 Elo but never measured with a working head-to-head pipeline.
- **ChessFormer-RL**: self-play PPO with sparse terminal rewards degraded from the SL initialization — the key negative result motivating the RL redesign.

## Models (legacy v1)

- [kaupane/ChessFormer-SL](https://huggingface.co/kaupane/ChessFormer-SL): SL checkpoint (~130k steps)
- [kaupane/ChessFormer-RL](https://huggingface.co/kaupane/ChessFormer-RL): RL initialization checkpoint (~50k steps)
