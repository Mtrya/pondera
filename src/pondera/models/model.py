"""Pondera's causal transformer with shared execution of consecutive blocks."""

from itertools import repeat

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from pondera.models.block_attn_res import DepthRead, summarize
from pondera.models.config import validate_model_config
from pondera.models.delta import KimiDeltaAttention
from pondera.models.move_head import FactorizedMoveHead
from pondera.models.navigation import navigation_sources, navigation_table
from pondera.models.softmax import CausalSelfAttention, RMSNorm, situ_glu
from pondera.models.temporal_navigation import NavigationRead


class Block(nn.Module):
    def __init__(self, width, intermediate_size, num_heads, num_kv_heads, attention):
        super().__init__()
        self.norm_attention = RMSNorm(width)
        mixer = CausalSelfAttention if attention == "softmax" else KimiDeltaAttention
        self.attention = mixer(width, num_heads, num_kv_heads)
        self.norm_ffn = RMSNorm(width)
        self.gate = nn.Linear(width, intermediate_size, bias=False)
        self.up = nn.Linear(width, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, width, bias=False)

    def feed_forward(self, x):
        x = self.norm_ffn(x)
        return self.down(situ_glu(self.gate(x), self.up(x)))


class PonderaModel(nn.Module):
    architecture = "pondera"

    def __init__(self, vocab_size, **settings):
        super().__init__()
        segments = validate_model_config(settings)
        self.vocab_size = vocab_size
        self.activation_checkpointing = False
        self.block_configs = tuple(
            block for segment in segments for block in segment.blocks
        )
        self.segment_sizes = tuple(len(segment.blocks) for segment in segments)
        self.repetitions = tuple(segment.repeat for segment in segments)
        self.loop_segments = tuple(
            index
            for index, segment in enumerate(settings["segments"])
            if "repeat" in segment
        )

        width = settings["hidden_size"]
        self.token_embedding = nn.Embedding(vocab_size, width)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        self.blocks = nn.ModuleList(
            Block(
                width,
                settings["intermediate_size"],
                settings["num_heads"],
                settings["num_kv_heads"],
                spec.attention,
            )
            for spec in self.block_configs
        )
        self.norm = RMSNorm(width)
        self.head = (
            FactorizedMoveHead(width, vocab_size)
            if settings.get("factorized_move_head", False)
            else nn.Linear(width, vocab_size, bias=False)
        )
        self.depth_reads = nn.ModuleList()
        if settings.get("block_attn_res", False):
            self.depth_reads.extend(
                DepthRead(width) for _ in range(2 * len(self.blocks))
            )
        self.navigation = nn.ModuleDict(
            {
                str(index): NavigationRead(width)
                for index, spec in enumerate(self.block_configs)
                if spec.navigation
            }
        )
        self._navigation_table = None
        if any(spec.value_source == "token_embedding" for spec in self.block_configs):
            self.register_buffer("value_rms_scale", torch.ones(width), persistent=False)
        else:
            self.value_rms_scale = None

    def override_repetitions(self, repetitions):
        if (
            type(repetitions) is not int
            or repetitions < 1
            or len(self.loop_segments) != 1
        ):
            raise ValueError(
                "A positive repetition override requires exactly one segment with an explicit repeat"
            )
        if repetitions > 1 and not self.depth_reads:
            raise ValueError("Repeated segments require Block AttnRes")
        index = self.loop_segments[0]
        self.repetitions = (
            *self.repetitions[:index],
            repetitions,
            *self.repetitions[index + 1 :],
        )

    def bind_tokenizer(self, dictionary):
        if self.navigation:
            self._navigation_table = navigation_table(dictionary)
            if len(self._navigation_table) != self.vocab_size:
                raise ValueError(
                    "Navigation dictionary must match the model vocabulary"
                )

    def prepare_navigation(self, tokens, boundaries):
        if self._navigation_table is None:
            raise ValueError("Bind the model tokenizer before using navigation")
        sources = navigation_sources(tokens, boundaries, self._navigation_table)
        receivers = np.flatnonzero(sources[:, 0] >= 0)
        return receivers, sources[receivers]

    def _token_values(self, embedding):
        if self.value_rms_scale is None:
            return None
        if self.depth_reads:
            embedding = embedding.float()
        eps = self.blocks[0].norm_attention.eps
        if (
            embedding.device.type == "npu"
            and embedding.dtype == self.value_rms_scale.dtype == torch.float32
        ):
            import torch_npu

            return torch_npu.npu_rms_norm(embedding, self.value_rms_scale, epsilon=eps)[
                0
            ]
        return (
            embedding
            * torch.rsqrt(embedding.float().square().mean(dim=-1, keepdim=True) + eps)
        ).to(embedding.dtype)

    def _read(self, position, residual, sources, partial_start, previous):
        if not self.depth_reads or position == 0:
            return residual
        if partial_start is not None:
            sources = (*sources, summarize(residual - partial_start))
        read = self.depth_reads[position - 1]
        if not previous:
            return read(residual, sources)
        earlier = sum(value for value, _ in previous)
        # Center the prior pass's contributions so a uniform read gives the current residual.
        return read(residual + earlier, (*sources, *previous)) - earlier

    def _block(
        self,
        index,
        residual,
        sources,
        group_start,
        has_partial_group,
        previous,
        token_values,
        prepared,
        cu_seqlens,
        cache,
        navigation_reads,
        host_tokens,
    ):
        block = self.blocks[index]
        spec = self.block_configs[index]
        attention_cache, navigation_cache = (None, None) if cache is None else cache
        partial_start = group_start if has_partial_group else None
        x = block.norm_attention(
            self._read(2 * index, residual, sources, partial_start, previous)
        )
        attention = block.attention
        if spec.value_source == "token_embedding":
            q, k, v = attention._project(x, value_input=token_values)
            if prepared is None:
                attended, attention_cache = attention.attend_projected_cached(
                    q, k, v, cu_seqlens, attention_cache
                )
            else:
                attended = attention.attend_projected(
                    q, k, v, prepared[type(attention)]
                )
        elif prepared is None:
            attended, attention_cache = attention.forward_cached(
                x, cu_seqlens, attention_cache
            )
        else:
            attended = attention(x, prepared[type(attention)])
        residual = residual + attended
        x = self._read(2 * index + 1, residual, sources, group_start, previous)
        residual = residual + block.feed_forward(x)
        if spec.navigation:
            navigation = self.navigation[str(index)]
            if prepared is None:
                residual, navigation_cache = navigation.forward_cached(
                    residual, host_tokens, self._navigation_table, navigation_cache
                )
            else:
                residual = navigation(residual, navigation_reads)
        return residual, (attention_cache, navigation_cache)

    def _execute(
        self,
        input_ids,
        cu_seqlens,
        *,
        cached=False,
        caches=None,
        navigation_reads=None,
        host_tokens=None,
    ):
        prepared = None
        if not cached:
            prepared = {
                attention: attention.prepare(cu_seqlens, input_ids.device)
                for attention in {type(block.attention) for block in self.blocks}
            }
        if caches is not None and len(caches) != sum(
            size * count
            for size, count in zip(self.segment_sizes, self.repetitions, strict=True)
        ):
            raise ValueError("Changing the execution layout requires a fresh cache")
        incoming = repeat(None) if caches is None else iter(caches)
        outgoing = []
        recompute = not cached and self.training and self.activation_checkpointing
        embedding = self.token_embedding(input_ids)
        token_values = self._token_values(embedding)
        residual = group_start = embedding
        sources = (summarize(embedding),) if self.depth_reads else ()
        offset = 0

        for size, count in zip(self.segment_sizes, self.repetitions, strict=True):
            entry = residual, group_start, sources
            previous = ()
            for pass_index in range(count):
                residual, group_start, sources = entry
                export = pass_index + 1 < count
                pass_group_start = residual
                deltas = []
                for index in range(offset, offset + size):
                    args = (
                        index,
                        residual,
                        sources,
                        group_start,
                        group_start is not residual,
                        previous,
                        token_values,
                        prepared,
                        cu_seqlens,
                        next(incoming),
                        navigation_reads,
                        host_tokens,
                    )
                    if recompute:
                        residual, cache = checkpoint(
                            self._block, *args, use_reentrant=False
                        )
                    else:
                        residual, cache = self._block(*args)
                    if cached:
                        outgoing.append(cache)
                    if self.depth_reads and self.block_configs[index].group_end:
                        group = summarize(residual - group_start)
                        sources = (*sources, group)
                        if export:
                            deltas.append(
                                group
                                if pass_group_start is group_start
                                else summarize(residual - pass_group_start)
                            )
                            pass_group_start = residual
                        group_start = residual
                # Split the exported deltas at the segment's own boundaries.
                # The outer residual keeps its original group start.
                if export and pass_group_start is not residual:
                    deltas.append(summarize(residual - pass_group_start))
                previous = tuple(deltas)
            offset += size

        partial_start = group_start if group_start is not residual else None
        x = self._read(2 * len(self.blocks), residual, sources, partial_start, ())
        return self.head(self.norm(x)), tuple(outgoing)

    def forward(self, input_ids, cu_seqlens, *, navigation_reads=None):
        if self.navigation and navigation_reads is None:
            navigation_reads = self.prepare_navigation(
                input_ids.detach().cpu().numpy(), cu_seqlens
            )
        return self._execute(input_ids, cu_seqlens, navigation_reads=navigation_reads)[
            0
        ]

    def forward_cached(self, input_ids, cu_seqlens, caches=None, *, host_tokens=None):
        cu_seqlens = torch.as_tensor(cu_seqlens, dtype=torch.int32)
        if self.navigation:
            if self._navigation_table is None:
                raise ValueError("Bind the model tokenizer before using navigation")
            if host_tokens is None:
                host_tokens = input_ids.detach().cpu().tolist()
        return self._execute(
            input_ids, cu_seqlens, cached=True, caches=caches, host_tokens=host_tokens
        )
