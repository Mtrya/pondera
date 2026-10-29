# Token backbones

`pondera.models` exposes three token-processing modules. `BaseBackbone` has a fixed stack with bidirectional attention. `SequenceBackbone` has a bidirectional board prefix and causal thought suffix. `DepthBackbone` refines a fixed token set through independently repeated cores. They return computed hidden states, repetition counts, and optional intermediate readouts. Inputs are embeddings; callers supply thought-position information together with thought embeddings.

```python
from pondera.models import BlockConfig, CoreConfig, DepthBackbone, RepetitionConfig

block = BlockConfig(hidden_size=256, intermediate_size=768, num_heads=8)
cores = [
    CoreConfig(layers=1),
    CoreConfig(layers=1, repetition=RepetitionConfig(max_repetitions=3)),
    CoreConfig(layers=2, repetition=RepetitionConfig(max_repetitions=2)),
    CoreConfig(layers=1),
]
model = DepthBackbone(block, cores)
output = model(embeddings, capture=True)
```

This schedule executes L1, then L2 three times, then the L3/L4 pair twice, then L5. Layers share weights only within their own repeated core. `capture=True` exposes states already computed after each layer; it adds no prediction network. Keep capture off when intermediate states are not needed. A head used only for supervision attaches outside the backbone. Deployment heads belong to the deployed model; the action/value heads in board policies are examples.

## Repetition

Each core has a `RepetitionConfig(max_repetitions=K)`: every valid token executes exactly K repetitions of that core before the next core starts. Padding tokens are excluded from attention and never counted. Because every valid token executes every repetition, computation stays dense over the input width; there is no early exit or sparse packing, so all K repetitions are always paid. `output.repetitions` reports the executed count per core, zero at padded positions.

## State access and decoding

Cache entries are keyed by physical core, physical layer, and repetition. Three access rules are available:

| Rule | Visible previous-token state at query repetition r |
|---|---|
| `matched_fallback` | The latest executed entry at or before r |
| `first` | The first repetition's entry at every repetition, including the current token's own key/value |
| `matched_only` | Only entries actually executed at r; other repetitions are masked out |

These rules apply during both parallel full-sequence computation and incremental decoding. They never expose a previous token's later repetition merely because it has already been computed in a sequential execution. A cache contains projected keys and values rather than a full hidden-state trace. Current storage uses tensors padded within each batch chunk; individual inference chunks retain only executed repetitions.

In evaluation mode, `prefill(board_embeddings)` returns a cache; `decode(new_embeddings, cache)` returns outputs for new tokens and an extended cache. The input cache is unchanged, allowing continuation from the same prefix. Supplying identical embedded inputs to full-sequence and cached execution should agree within numerical tolerance. The board prefix cannot attend to thoughts, and thoughts cannot attend to future thoughts. Sequence termination is handled by a designated token in the eventual thought vocabulary, not a separate sequence-length controller.

## Board policies

`BoardPolicyModel` combines `BoardEmbedding`, `BaseBackbone`, and shared action/value readouts. `DepthPolicyModel` uses the same board and output interfaces with `DepthBackbone`; its shared vocabulary head is read at every core exit, and the final exit is the deployed answer. Board inputs use the compact-piece layout with joint (square, piece) embeddings; black-to-move positions are canonicalized by rank reflection, and logits are reflected back to the absolute orientation. The action readout is a single token over the full move vocabulary. The representation study behind these choices is summarized in [board representation](research/board-representation.md) and [move readout](research/move-readout.md).

`pondera.training.cost.policy_flops` defines the dense-matmul accounting convention used by the trainer, including attention, projections, FFNs, and the action/value projections. Sequence token vocabulary, feedback, termination-symbol encoding, and supervision objectives are deliberately not specified by these modules.
