"""Portable SDPA blocks with active-query packing."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from pondera.models.cache import KVState
from pondera.models.config import BlockConfig


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


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward,
        dropout: float,
        bias_up: bool = False,
        bias_gate: bool = False,
        bias_down: bool = True,
        dtype=None,
    ):
        super().__init__()
        self.up_proj = nn.Linear(d_model, dim_feedforward, bias=bias_up, dtype=dtype)
        self.gate_proj = nn.Linear(
            d_model, dim_feedforward, bias=bias_gate, dtype=dtype
        )
        self.down_proj = nn.Linear(
            dim_feedforward, d_model, bias=bias_down, dtype=dtype
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.up_proj(x) * self.dropout(F.silu(self.gate_proj(x)))
        return self.down_proj(x)


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
        self.attention_capture = None

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
        key = kv.new_zeros(shape).scatter(2, index, kv[0])
        value = kv.new_zeros(shape).scatter(2, index, kv[1])
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
    ):
        batch, queries, width = normalized.shape
        q = (
            self.query(normalized)
            .view(batch, queries, self.heads, self.head_size)
            .transpose(1, 2)
        )
        key, value = memory.key, memory.value
        mask = attention_mask(
            query_positions, memory.valid.shape[1], prefix_length, memory.valid
        )
        if self.attention_capture is not None:
            self._capture_attention(q, key, mask, layout.valid)
        attention = (
            F.scaled_dot_product_attention(
                q,
                key,
                value,
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

    @torch.no_grad()
    def _capture_attention(self, queries, keys, mask, query_valid):
        """Per-head attention entropy and peak weight; probe-only path."""
        scores = queries.float() @ keys.float().transpose(-1, -2)
        scores = scores / math.sqrt(self.head_size)
        scores = scores.masked_fill(~mask, float("-inf"))
        probs = scores.softmax(-1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
        peak = probs.max(-1).values
        rows = query_valid[:, None, :] & mask.any(-1)
        if rows.shape[1] == 1:
            rows = rows.expand_as(entropy)
        stats = []
        for head in range(self.heads):
            selected = rows[:, head]
            if not selected.any():
                continue
            stats.append(
                (
                    float(entropy[:, head][selected].mean()),
                    float(peak[:, head][selected].mean()),
                )
            )
        self.attention_capture.append(stats)


def sinusoidal_positions(positions, width, dtype):
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=positions.device, dtype=torch.float32)
        * (-math.log(10000.0) / width)
    )
    angles = positions.float().unsqueeze(-1) * frequencies
    result = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)[..., :width]
    return result.to(dtype)
