# Token backbones

`pondera.models` exposes four token-processing modules. `BaseBackbone` has a fixed stack with bidirectional attention. `SequenceBackbone` has a bidirectional board prefix and causal thought suffix. `DepthBackbone` refines a fixed token set through independently repeated cores. `CombinedBackbone` combines causal sequence expansion with repeated cores. They return computed hidden states, repetition counts, and optional intermediate readouts. Inputs are embeddings; callers supply thought-position information together with thought embeddings.

```python
from pondera.models import BlockConfig, CombinedBackbone, CoreConfig, RepetitionConfig

block = BlockConfig(hidden_size=256, intermediate_size=768, num_heads=8)
cores = [
    CoreConfig(layers=1),
    CoreConfig(layers=1, repetition=RepetitionConfig(max_repetitions=3)),
    CoreConfig(layers=2, repetition=RepetitionConfig(max_repetitions=2)),
    CoreConfig(layers=1),
]
model = CombinedBackbone(block, cores, cache_policy="matched_fallback")
output = model(embeddings, prefix_length=board_length, capture=True)
```

This schedule executes L1, then L2 three times, then the L3/L4 pair twice, then L5. Layers share weights only within their own repeated core. `capture=True` exposes states already computed after each layer; it adds no prediction network. Keep capture off when intermediate states are not needed. A head used only for supervision attaches outside the backbone. Deployment heads belong to the deployed model; the action/value heads in board policies and learned repetition controllers are examples.

## Repetition

Each core has a `RepetitionConfig`: `mode` is `fixed`, `convergence`, or `learned`; `min_repetitions` and `max_repetitions` bound execution. Fixed mode uses the maximum or explicit per-token limits. Convergence compares the relative RMS change in a token's hidden state after a completed core repetition. Learned mode applies RMS normalization and a linear stop head. Its hard threshold controls execution, while differentiable logits and active masks are returned in `output.controllers` for an external training objective. Hard decisions do not provide a gradient through the number of repetitions by themselves.

`scope="individual"` assigns each token its own decision. `scope="shared"` shares decisions across all valid tokens in a depth backbone, or only across the board prefix in a combined backbone. Thought tokens in a combined backbone always decide individually. A shared learned controller averages normalized group states; shared convergence requires every group member to satisfy the tolerance. Padding never contributes. Optional `repetition_limits` supply one `(batch, tokens)` int64 hard-cap tensor per core, within that core's configured bounds; members of a shared group must have identical limits.

`reinject_input=True` adds a learned projection of that core's entry state before every repetition, including the first. `condition_on_iteration=True` adds a learned embedding for the current repetition within the configured limit. These options are independent per core. Active queries are packed by batch and exited tokens skip subsequent attention queries and FFNs. Packing still pads to the largest active count in a batch and introduces scheduling overhead; it is not a claim of optimal sparse execution.

## State access and decoding

Cache entries are keyed by physical core, physical layer, and repetition. Three access rules are available:

| Rule | Visible previous-token state at query repetition r |
|---|---|
| `matched_fallback` | The latest executed entry at or before r |
| `first` | The first repetition's entry at every repetition, including the current token's own key/value |
| `matched_only` | Only entries actually executed at r; earlier exits are masked out |

These rules apply during both parallel full-sequence computation and incremental decoding. They never expose a previous token's later repetition merely because it has already been computed in a sequential execution. Thus a token that exits at repetition 2 contributes entries 1, 2, 2, 2 under fallback when another token executes four repetitions. A cache contains projected keys and values rather than a full hidden-state trace. Current storage uses tensors padded within each batch chunk; individual inference chunks retain only executed repetitions.

In evaluation mode, `prefill(board_embeddings)` returns a cache; `decode(new_embeddings, cache)` returns outputs for new tokens and an extended cache. The input cache is unchanged, allowing continuation from the same prefix. Supplying identical embedded inputs to full-sequence and cached execution should agree within numerical tolerance. The board prefix cannot attend to thoughts, and thoughts cannot attend to future thoughts. Relative board geometry never overrides these masks. Sequence termination is handled by a designated token in the eventual thought vocabulary, not a separate sequence-length controller.

## Board policy comparison

`BoardPolicyModel` combines `BoardEmbedding`, `BaseBackbone`, and shared action/value readouts. `DepthPolicyModel` uses the same board and output interfaces with `DepthBackbone`. The base policy supports square or compact-piece layouts, both preserving exact source-square identity and the same metadata. Absolute geometry adds square embeddings. Relative geometry retains those embeddings and additionally learns one scalar attention bias per head and signed rank/file displacement between board squares. Metadata and readout tokens receive no board-relative bias.

Use the base policy to compare representations before transferring the selected input to all three adaptive variants. Match trainable parameter count and theoretical training FLOPs. Do not add unused parameters merely to match a count; state any small residual mismatch after adjusting FFN width. Runtime, throughput, and memory remain optimization diagnostics. `pondera.training.cost.policy_flops` defines the dense-matmul accounting convention used by the trainer, including attention, projections, FFNs, and the action/value projections. Sequence token vocabulary, feedback, termination-symbol encoding, and supervision objectives are deliberately not specified by these modules.
