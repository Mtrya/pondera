"""Base, causal sequence, recurrent depth, and combined token backbones.

Inputs are embeddings; token meaning and training-only prediction heads belong
to callers. Causal callers also supply sequence-position information in those
embeddings.
"""

from dataclasses import dataclass, field

import torch
from torch import nn

from pondera.models.blocks import QueryLayout, TokenBlock
from pondera.models.cache import (
    CACHE_POLICIES,
    BackboneCache,
    join_states,
    select_state,
)
from pondera.models.config import BlockConfig, CoreConfig
from pondera.models.controllers import make_controller


@dataclass
class Readout:
    core: int
    layer: int
    repetition: int
    hidden: torch.Tensor
    active: torch.Tensor


@dataclass
class ControllerReadout:
    core: int
    repetition: int
    logits: torch.Tensor
    active: torch.Tensor


@dataclass
class BackboneOutput:
    hidden: torch.Tensor
    repetitions: tuple[torch.Tensor, ...]
    readouts: tuple[Readout, ...] = ()
    controllers: tuple[ControllerReadout, ...] = ()
    cache: BackboneCache | None = None


class RepeatedCore(nn.Module):
    def __init__(self, block: BlockConfig, config: CoreConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(TokenBlock(block) for _ in range(config.layers))
        self.controller = make_controller(config.repetition, block.hidden_size)
        self.reinjection = (
            nn.Linear(block.hidden_size, block.hidden_size, bias=False)
            if config.reinject_input
            else None
        )
        self.iteration = (
            nn.Embedding(config.repetition.max_repetitions, block.hidden_size)
            if config.condition_on_iteration
            else None
        )


@dataclass
class Execution:
    valid: torch.Tensor
    prefix_length: int | None
    offset: int = 0
    past: BackboneCache | None = None
    capture: bool = False
    save_cache: bool = False
    readouts: list = field(default_factory=list)
    controllers: list = field(default_factory=list)
    histories: dict = field(default_factory=dict)


class TokenBackbone(nn.Module):
    """Shared execution primitives, composed by the four public backbones."""

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

    def _core(self, core, index, x, execution, limits):
        config = core.config.repetition
        valid = execution.valid
        if limits is None:
            limits = torch.full_like(valid, config.max_repetitions, dtype=torch.long)
            variable_limits = False
        else:
            if limits.shape != valid.shape or limits.dtype != torch.long:
                raise ValueError(
                    "Repetition limits must be int64 with shape (batch, tokens)"
                )
            if bool(
                ((limits < config.min_repetitions) | (limits > config.max_repetitions))[
                    valid
                ].any()
            ):
                raise ValueError("Repetition limits lie outside configured bounds")
            variable_limits = True
        shared = torch.zeros_like(valid)
        if config.scope == "shared":
            if execution.prefix_length is None:
                shared = valid
            else:
                positions = torch.arange(x.shape[1], device=x.device) + execution.offset
                shared = valid & (positions[None, :] < execution.prefix_length)
            if variable_limits:
                low = torch.where(shared, limits, config.max_repetitions).amin(1)
                high = torch.where(shared, limits, config.min_repetitions).amax(1)
                if bool(((low != high) & shared.any(1)).any()):
                    raise ValueError(
                        "Members of a shared decision need identical limits"
                    )
        entry = x
        active = valid
        counts = torch.zeros_like(valid, dtype=torch.long)
        histories = [[] for _ in core.layers]
        for repetition in range(config.max_repetitions):
            if repetition == 0 or (config.mode == "fixed" and not variable_limits):
                layout = QueryLayout.full(active)
            else:
                layout = QueryLayout.from_mask(active)
            if layout.indices.shape[1] == 0:
                break
            previous = x
            if core.reinjection is not None:
                injected = layout.gather(x) + core.reinjection(layout.gather(entry))
                x = layout.scatter(injected, x)
            if core.iteration is not None:
                conditioned = layout.gather(x) + core.iteration.weight[repetition]
                x = layout.scatter(conditioned, x)
            query_positions = layout.indices + execution.offset
            for layer_index, layer in enumerate(core.layers):
                normalized = layer.norm_attention(layout.gather(x))
                history = histories[layer_index]
                # First-repetition sharing does not recompute later K/V projections.
                if repetition == 0 or self.cache_policy != "first":
                    history.append(layer.project(normalized, layout))
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
                    layout,
                    join_states(states),
                    query_positions,
                    execution.prefix_length,
                )
                if execution.capture:
                    execution.readouts.append(
                        Readout(index, layer_index, repetition + 1, x, active)
                    )
            counts = counts + active.long()
            decision = core.controller(previous, x, shared)
            if decision.logits is not None:
                execution.controllers.append(
                    ControllerReadout(index, repetition + 1, decision.logits, active)
                )
            can_stop = repetition + 1 >= config.min_repetitions
            active = active & (counts < limits) & ~(decision.stop & can_stop)
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
        return x, counts

    def _run(
        self,
        x,
        *,
        valid=None,
        prefix_length=None,
        cache=None,
        return_cache=False,
        capture=False,
        repetition_limits=None,
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
        if repetition_limits is None:
            repetition_limits = [None] * len(self.cores)
        if len(repetition_limits) != len(self.cores):
            raise ValueError("Supply one repetition-limit tensor per core")
        execution = Execution(
            valid,
            prefix_length,
            0 if cache is None else cache.length,
            cache,
            capture,
            return_cache,
        )
        counts = []
        for i, (core, limits) in enumerate(zip(self.cores, repetition_limits)):
            x, count = self._core(core, i, x, execution, limits)
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
            tuple(execution.controllers),
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

    def forward(
        self, tokens, *, valid=None, capture=False, repetition_limits=None
    ):
        return self._run(
            tokens,
            valid=valid,
            capture=capture,
            repetition_limits=repetition_limits,
        )


class CausalBackbone(TokenBackbone):
    """Bidirectional board prefix followed by causally attended thought tokens."""

    def forward(
        self,
        tokens,
        prefix_length,
        *,
        valid=None,
        capture=False,
        repetition_limits=None,
        return_cache=False,
    ):
        if not 0 <= prefix_length <= tokens.shape[1]:
            raise ValueError("Invalid board prefix length")
        return self._run(
            tokens,
            valid=valid,
            prefix_length=prefix_length,
            capture=capture,
            repetition_limits=repetition_limits,
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
        repetition_limits=None,
    ):
        return self._run(
            tokens,
            valid=valid,
            prefix_length=cache.prefix_length,
            cache=cache,
            return_cache=True,
            capture=capture,
            repetition_limits=repetition_limits,
        )


class SequenceBackbone(CausalBackbone):
    """Fixed-depth causal sequence expansion with prefix KV caching."""

    def __init__(self, block: BlockConfig, num_layers: int):
        super().__init__(block, [CoreConfig(layers=num_layers)], "matched_fallback")


class CombinedBackbone(CausalBackbone):
    """Causal sequence expansion with adaptive per-thought recurrent depth.

    Shared repetition decisions, when configured, apply only to board-prefix
    tokens. Every thought token always receives its own stopping decision.
    """

    def __init__(
        self,
        block: BlockConfig,
        cores: list[CoreConfig],
        cache_policy="matched_fallback",
    ):
        super().__init__(block, cores, cache_policy)
