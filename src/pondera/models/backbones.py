"""Base, causal sequence, and recurrent depth token backbones.

Inputs are embeddings; token meaning and training-only prediction heads belong
to callers. Causal callers also supply sequence-position information in those
embeddings.
"""

from dataclasses import dataclass, field

import torch
from torch import nn

from pondera.models.blocks import TokenBlock
from pondera.models.cache import (
    CACHE_POLICIES,
    BackboneCache,
    join_states,
    select_state,
)
from pondera.models.config import BlockConfig, CoreConfig


@dataclass
class Readout:
    core: int
    layer: int
    repetition: int
    hidden: torch.Tensor
    active: torch.Tensor


@dataclass
class BackboneOutput:
    hidden: torch.Tensor
    repetitions: tuple[torch.Tensor, ...]
    readouts: tuple[Readout, ...] = ()
    cache: BackboneCache | None = None


class RepeatedCore(nn.Module):
    def __init__(self, block: BlockConfig, config: CoreConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(TokenBlock(block) for _ in range(config.layers))


@dataclass
class Execution:
    valid: torch.Tensor
    prefix_length: int | None
    offset: int = 0
    past: BackboneCache | None = None
    capture: bool = False
    save_cache: bool = False
    readouts: list = field(default_factory=list)
    histories: dict = field(default_factory=dict)


class TokenBackbone(nn.Module):
    """Shared execution primitives, composed by the three public backbones."""

    def __init__(self, block, cores, cache_policy):
        super().__init__()
        if cache_policy not in CACHE_POLICIES or not cores:
            raise ValueError(
                "Need a nonempty core schedule and a supported cache policy"
            )
        self.block_config = block
        self.cache_policy = cache_policy
        self.cores = nn.ModuleList(RepeatedCore(block, core) for core in cores)
        self.final_norm = nn.RMSNorm(block.hidden_size)

    def _core(self, core, index, x, execution):
        config = core.config.repetition
        valid = execution.valid
        query_positions = (
            torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
            + execution.offset
        )
        histories = [[] for _ in core.layers]
        for repetition in range(config.max_repetitions):
            for layer_index, layer in enumerate(core.layers):
                normalized = layer.norm_attention(x)
                history = histories[layer_index]
                # First-repetition sharing does not recompute later K/V projections.
                if repetition == 0 or self.cache_policy != "first":
                    history.append(layer.project(normalized, valid))
                states = []
                if execution.past is not None:
                    states.extend(
                        select_state(chunk, repetition, self.cache_policy)
                        for chunk in execution.past.histories[index, layer_index]
                    )
                states.append(select_state(history, repetition, self.cache_policy))
                x = layer.attend(
                    x,
                    normalized,
                    valid,
                    join_states(states),
                    query_positions,
                    execution.prefix_length,
                )
                if execution.capture:
                    execution.readouts.append(
                        Readout(index, layer_index, repetition + 1, x, valid)
                    )
        if execution.save_cache:
            for layer_index, history in enumerate(histories):
                past = (
                    []
                    if execution.past is None
                    else execution.past.histories[index, layer_index]
                )
                execution.histories[index, layer_index] = past + [
                    [state.detach() for state in history]
                ]
        return x, valid.long() * config.max_repetitions

    def _run(
        self,
        x,
        *,
        valid=None,
        prefix_length=None,
        cache=None,
        return_cache=False,
        capture=False,
    ):
        if (
            x.ndim != 3
            or x.shape[-1] != self.block_config.hidden_size
            or not x.shape[1]
        ):
            raise ValueError(
                "Expected nonempty (batch, tokens, hidden_size) embeddings"
            )
        if valid is None:
            valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if valid.shape != x.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("Invalid token mask")
        if cache is not None:
            first = next(iter(cache.histories.values()))[0][0]
            if cache.owner != id(self) or first.key.shape[0] != x.shape[0]:
                raise ValueError("Cache belongs to another backbone or batch")
            if cache.prefix_length != prefix_length:
                raise ValueError("The board prefix cannot change during decoding")
        if return_cache and self.training:
            raise ValueError("Incremental caches require evaluation mode")
        execution = Execution(
            valid,
            prefix_length,
            0 if cache is None else cache.length,
            cache,
            capture,
            return_cache,
        )
        counts = []
        for i, core in enumerate(self.cores):
            x, count = self._core(core, i, x, execution)
            counts.append(count)
        output_cache = None
        if return_cache:
            output_cache = BackboneCache(
                id(self),
                prefix_length,
                execution.offset + x.shape[1],
                execution.histories,
            )
        hidden = self.final_norm(x).masked_fill(~valid.unsqueeze(-1), 0)
        return BackboneOutput(
            hidden,
            tuple(counts),
            tuple(execution.readouts),
            output_cache,
        )


class BaseBackbone(TokenBackbone):
    """A fixed-depth, fully bidirectional transformer."""

    def __init__(self, block: BlockConfig, num_layers: int):
        super().__init__(block, [CoreConfig(layers=num_layers)], "matched_fallback")

    def forward(self, tokens, *, valid=None, capture=False):
        return self._run(tokens, valid=valid, capture=capture)


class DepthBackbone(TokenBackbone):
    """Bidirectional tokens refined by independently configured repeated cores."""

    def __init__(
        self,
        block: BlockConfig,
        cores: list[CoreConfig],
        cache_policy="matched_fallback",
    ):
        super().__init__(block, cores, cache_policy)

    def forward(self, tokens, *, valid=None, capture=False):
        return self._run(tokens, valid=valid, capture=capture)


class CausalBackbone(TokenBackbone):
    """Bidirectional board prefix followed by causally attended thought tokens."""

    def forward(
        self,
        tokens,
        prefix_length,
        *,
        valid=None,
        capture=False,
        return_cache=False,
    ):
        if not 0 <= prefix_length <= tokens.shape[1]:
            raise ValueError("Invalid board prefix length")
        return self._run(
            tokens,
            valid=valid,
            prefix_length=prefix_length,
            capture=capture,
            return_cache=return_cache,
        )

    def prefill(self, board_tokens, **kwargs):
        return self.forward(
            board_tokens, board_tokens.shape[1], return_cache=True, **kwargs
        )

    def decode(
        self,
        tokens,
        cache: BackboneCache,
        *,
        valid=None,
        capture=False,
    ):
        return self._run(
            tokens,
            valid=valid,
            prefix_length=cache.prefix_length,
            cache=cache,
            return_cache=True,
            capture=capture,
        )


class SequenceBackbone(CausalBackbone):
    """Fixed-depth causal sequence expansion with prefix KV caching."""

    def __init__(self, block: BlockConfig, num_layers: int):
        super().__init__(block, [CoreConfig(layers=num_layers)], "matched_fallback")
