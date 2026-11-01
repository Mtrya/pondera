"""Fixed-depth causal Transformer with a bidirectional board prefix."""

from dataclasses import dataclass

import torch
from torch import nn

from pondera.models.blocks import TokenBlock
from pondera.models.cache import BackboneCache, KVState, join_states
from pondera.models.config import BlockConfig


@dataclass
class BackboneOutput:
    hidden: torch.Tensor
    cache: BackboneCache | None = None


class Backbone(nn.Module):
    def __init__(self, block: BlockConfig, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("A trace backbone needs at least one layer")
        self.block_config = block
        self.layers = nn.ModuleList(TokenBlock(block) for _ in range(num_layers))
        self.final_norm = nn.RMSNorm(block.hidden_size)

    def _run(self, x, prefix_length, *, valid=None, cache=None, return_cache=False):
        if (
            x.ndim != 3
            or x.shape[-1] != self.block_config.hidden_size
            or not x.shape[1]
        ):
            raise ValueError(
                "Expected nonempty (batch, tokens, hidden_size) embeddings"
            )
        if not 0 <= prefix_length <= (cache.prefix_length if cache else x.shape[1]):
            raise ValueError("Invalid board prefix length")
        if valid is None:
            valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if valid.shape != x.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("Invalid token mask")
        if cache is not None:
            if cache.owner != id(self) or cache.prefix_length != prefix_length:
                raise ValueError("Cache belongs to another backbone or prefix")
            if any(
                states[0].key.shape[0] != x.shape[0]
                for states in cache.histories.values()
            ):
                raise ValueError("Cache belongs to another batch")
        if return_cache and self.training:
            raise ValueError("Incremental caches require evaluation mode")

        offset = 0 if cache is None else cache.length
        query_positions = (
            torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1) + offset
        )
        histories: dict[int, list[KVState]] = {}
        for index, layer in enumerate(self.layers):
            past = [] if cache is None else cache.histories[index]
            normalized = layer.norm_attention(x)
            current = layer.project(normalized, valid)
            x = layer.attend(
                x,
                normalized,
                valid,
                join_states([*past, current]),
                query_positions,
                prefix_length,
            )
            if return_cache:
                histories[index] = [*past, current.detach()]
        hidden = self.final_norm(x).masked_fill(~valid.unsqueeze(-1), 0)
        output_cache = (
            BackboneCache(id(self), prefix_length, offset + x.shape[1], histories)
            if return_cache
            else None
        )
        return BackboneOutput(hidden, output_cache)

    def forward(self, tokens, prefix_length, *, valid=None):
        if not 0 <= prefix_length <= tokens.shape[1]:
            raise ValueError("Invalid board prefix length")
        return self._run(tokens, prefix_length, valid=valid)

    def prefill(self, board_tokens, *, valid=None):
        return self._run(
            board_tokens,
            board_tokens.shape[1],
            valid=valid,
            return_cache=True,
        )

    def decode(self, tokens, cache: BackboneCache, *, valid=None):
        return self._run(
            tokens,
            cache.prefix_length,
            valid=valid,
            cache=cache,
            return_cache=True,
        )
