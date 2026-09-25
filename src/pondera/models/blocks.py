"""Portable SDPA blocks with optional board geometry and active-query packing."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from pondera.models.cache import KVState
from pondera.models.config import BlockConfig
from pondera.models.transformer import SwiGLUFFN


@dataclass
class QueryLayout:
    indices: torch.Tensor
    valid: torch.Tensor
    full_length: int
    dense: bool = False

    @classmethod
    def from_mask(cls, active):
        counts = active.sum(1)
        width = int(counts.max())
        indices = torch.argsort(~active, dim=1, stable=True)[:, :width]
        valid = torch.arange(width, device=active.device)[None, :] < counts[:, None]
        return cls(indices, valid, active.shape[1])

    @classmethod
    def full(cls, valid):
        indices = torch.arange(valid.shape[1], device=valid.device).expand(
            valid.shape[0], -1
        )
        return cls(indices, valid, valid.shape[1], dense=True)

    def gather(self, values):
        if self.dense:
            return values
        return values.gather(
            1, self.indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        )

    def scatter(self, values, original):
        if self.dense:
            return torch.where(self.valid.unsqueeze(-1), values, original)
        previous = self.gather(original)
        updates = torch.where(self.valid.unsqueeze(-1), values, previous)
        return original.scatter(
            1, self.indices.unsqueeze(-1).expand_as(updates), updates
        )


def attention_mask(query_positions, key_length, prefix_length, key_valid):
    """Boolean SDPA mask: True means accessible."""
    keys = torch.arange(key_length, device=query_positions.device)[None, None, :]
    queries = query_positions[:, :, None]
    if prefix_length is None:
        accessible = torch.ones_like(queries + keys, dtype=torch.bool)
    else:
        accessible = ((queries < prefix_length) & (keys < prefix_length)) | (
            (queries >= prefix_length) & (keys <= queries)
        )
    return accessible[:, None, :, :] & key_valid[:, None, None, :]


class RelativeBoardBias(nn.Module):
    """A learned bias per head and signed (file, rank) displacement."""

    def __init__(self, heads):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.zeros(15 * 15, heads), freeze=False
        )

    def forward(self, query_squares, key_squares):
        q, k = query_squares[:, :, None], key_squares[:, None, :]
        pair = (q >= 0) & (k >= 0)
        relative = (
            (k.clamp_min(0) // 8 - q.clamp_min(0) // 8 + 7) * 15
            + k.clamp_min(0) % 8
            - q.clamp_min(0) % 8
            + 7
        )
        bias = self.embedding(relative).permute(0, 3, 1, 2)
        return bias * pair[:, None, :, :]


class TokenBlock(nn.Module):
    def __init__(self, config: BlockConfig):
        super().__init__()
        self.heads = config.num_heads
        self.head_size = config.hidden_size // config.num_heads
        self.dropout = config.dropout
        self.norm_attention = nn.RMSNorm(config.hidden_size)
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key_value = nn.Linear(
            config.hidden_size, 2 * config.hidden_size, bias=False
        )
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.norm_ffn = nn.RMSNorm(config.hidden_size)
        self.ffn = SwiGLUFFN(
            config.hidden_size, config.intermediate_size, config.dropout
        )
        self.geometry = (
            RelativeBoardBias(config.num_heads)
            if config.geometry == "relative"
            else None
        )

    def project(self, normalized, layout: QueryLayout):
        batch, queries, width = normalized.shape
        kv = self.key_value(normalized).view(
            batch, queries, 2, self.heads, self.head_size
        )
        kv = kv.permute(2, 0, 3, 1, 4)
        if layout.dense:
            return KVState(kv[0], kv[1], layout.valid)
        index = layout.indices[:, None, :, None].expand(
            -1, self.heads, -1, self.head_size
        )
        shape = (batch, self.heads, layout.full_length, self.head_size)
        key = normalized.new_zeros(shape).scatter(2, index, kv[0])
        value = normalized.new_zeros(shape).scatter(2, index, kv[1])
        valid = torch.zeros(
            batch, layout.full_length, device=normalized.device, dtype=torch.bool
        )
        valid = valid.scatter(1, layout.indices, layout.valid)
        return KVState(key, value, valid)

    def attend(
        self,
        x,
        normalized,
        layout,
        memory,
        query_positions,
        prefix_length,
        query_squares,
        key_squares,
    ):
        batch, queries, width = normalized.shape
        q = (
            self.query(normalized)
            .view(batch, queries, self.heads, self.head_size)
            .transpose(1, 2)
        )
        mask = attention_mask(
            query_positions, memory.valid.shape[1], prefix_length, memory.valid
        )
        if self.geometry is not None:
            bias = self.geometry(query_squares, key_squares).to(q.dtype)
            mask = bias.masked_fill(~mask, float("-inf"))
        attention = (
            F.scaled_dot_product_attention(
                q,
                memory.key,
                memory.value,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
            .transpose(1, 2)
            .reshape(batch, queries, width)
        )
        updated = layout.gather(x) + F.dropout(
            self.output(attention), self.dropout, self.training
        )
        updated = updated + F.dropout(
            self.ffn(self.norm_ffn(updated)), self.dropout, self.training
        )
        return layout.scatter(updated, x)


def sinusoidal_positions(positions, width, dtype):
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=positions.device, dtype=torch.float32)
        * (-math.log(10000.0) / width)
    )
    angles = positions.float().unsqueeze(-1) * frequencies
    result = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)[..., :width]
    return result.to(dtype)
