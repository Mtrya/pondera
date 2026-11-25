"""Block Attention Residuals with the original packed KDA and token-value paths."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from pondera.models.token_value import KimiDeltaTokenValueModel
from pondera.models.vanilla import CausalSelfAttention


def summarize(value):
    value = value.float()
    inverse_rms = torch.rsqrt(value.square().mean(dim=-1) + 1e-6)
    return value, inverse_rms


class DepthRead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(width))
        self.key_gain = nn.Parameter(torch.ones(width))

    def forward(self, residual, sources):
        query = (self.query * self.key_gain).float().unsqueeze(0)
        if self.training and query.device.type == "npu" and len(sources) <= 5:
            from pondera.models._depth_read_triton import depth_read

            # The group increments telescope to the residual. Scaling the mean by
            # the number of sources gives the same read with a fused backward.
            return len(sources) * depth_read(sources, query)
        with torch.autocast(query.device.type, enabled=False):
            scores = torch.stack(
                [
                    F.linear(value, query).squeeze(-1) * inverse_rms
                    for value, inverse_rms in sources
                ],
                dim=-1,
            )
        weights = (len(sources) * scores.softmax(dim=-1) - 1).unbind(-1)
        # Unit weights reconstruct the trained residual with its original addition order.
        for (value, _), weight in zip(sources, weights, strict=True):
            residual = torch.addcmul(residual, value, weight.unsqueeze(-1))
        return residual


class TokenValueBlockAttnResModel(KimiDeltaTokenValueModel):
    """Each completed pair of Transformer blocks supplies one depth source."""

    architecture = "kda_token_value_block_attn_res"

    def __init__(self, *, hidden_size, **settings):
        super().__init__(hidden_size=hidden_size, **settings)
        self.depth_reads = nn.ModuleList(
            DepthRead(hidden_size) for _ in range(2 * len(self.blocks))
        )

    def _read(self, residual, sources, group_start, position):
        if position == 0:
            return residual
        if group_start is not None:
            sources = (*sources, summarize(residual - group_start))
        return self.depth_reads[position - 1](residual, sources)

    def _block(
        self,
        index,
        residual,
        group_start,
        sources,
        prepared=None,
        cu_seqlens=None,
        cache=None,
    ):
        block = self.blocks[index]
        partial_start = group_start if index % 2 else None
        reader_input = self._read(residual, sources, partial_start, 2 * index)
        next_cache = None
        if index + 1 == len(self.blocks):
            q, k, v = self._project(block, reader_input, sources[0][0])
            if cu_seqlens is None:
                attended = block.attention.attend_projected(
                    q, k, v, prepared[CausalSelfAttention]
                )
            else:
                attended, next_cache = block.attention.attend_projected_cached(
                    q, k, v, cu_seqlens, cache
                )
        else:
            normalized = block.norm_attention(reader_input)
            if cu_seqlens is None:
                attended = block.attention(normalized, prepared[type(block.attention)])
            else:
                attended, next_cache = block.attention.forward_cached(
                    normalized, cu_seqlens, cache
                )
        residual = residual + attended
        reader_input = self._read(residual, sources, group_start, 2 * index + 1)
        return residual + block._ffn(block.norm_ffn(reader_input)), next_cache

    def _run(
        self, input_ids, cu_seqlens, caches=None, *, cached=False, after_block=None
    ):
        prepared = None if cached else self.prepare(cu_seqlens, input_ids.device)
        group_start = residual = self.token_embedding(input_ids)
        sources = (summarize(residual),)
        new_caches = []
        for index in range(len(self.blocks)):
            args = (index, residual, group_start, sources)
            if cached:
                cache = None if caches is None else caches[index]
                residual, next_cache = self._block(
                    *args, cu_seqlens=cu_seqlens, cache=cache
                )
                new_caches.append(next_cache)
            elif self.training and self.activation_checkpointing:
                residual, _ = checkpoint(
                    self._block, *args, prepared, use_reentrant=False
                )
            else:
                residual, _ = self._block(*args, prepared)
            if after_block is not None:
                residual = after_block(index, residual)
            if index % 2 == 1 or index + 1 == len(self.blocks):
                sources = (*sources, summarize(residual - group_start))
                group_start = residual
        result = self._read(residual, sources, None, 2 * len(self.blocks))
        return self.head(self.norm(result)), new_caches

    def forward(self, input_ids, cu_seqlens, *, after_block=None):
        return self._run(input_ids, cu_seqlens, after_block=after_block)[0]

    def forward_cached(self, input_ids, cu_seqlens, caches=None, *, after_block=None):
        return self._run(
            input_ids, cu_seqlens, caches, cached=True, after_block=after_block
        )
