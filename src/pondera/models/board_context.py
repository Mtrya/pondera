"""Causal root-board conditioning and fixed chess circuits for attention."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from pondera.chess.vocabulary import BOARD_PIECES, BOARD_TOKENS, MOVE
from pondera.models.delta import GatedDeltaAttention, KimiDeltaAttention
from pondera.models.memory import RecurrentAttention
from pondera.models.vanilla import (
    CausalSelfAttention,
    RMSNorm,
    VanillaModel,
    varlen_attention,
)

# At most 32 occupied squares followed by eight metadata atoms.
PREFIX_LENGTH = 40
CONTEXT_DIM = 12 + 2 * 64


class LearnedBoardFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(MOVE, CONTEXT_DIM)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, totals):
        return totals


class CompiledBoardFeatures(nn.Module):
    """Piece counts and attack counts, compiled into frozen affine/ReLU layers.

    For each ray, relu(slider_at_origin - occupied_between) is its exact
    attack indicator. Target occupancy does not block an attack on that target.
    These are geometric attacks, including attacks by pinned pieces.
    """

    def __init__(self):
        super().__init__()
        embedding = torch.zeros(MOVE, BOARD_TOKENS)
        embedding[:BOARD_TOKENS] = torch.eye(BOARD_TOKENS)
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
        direct = torch.zeros(CONTEXT_DIM, BOARD_TOKENS)
        rays, destinations = [], []
        directions = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
        for square in range(64):
            rank, file = divmod(square, 8)
            for piece in range(BOARD_PIECES):
                direct[piece, square * BOARD_PIECES + piece] = 1
                color, kind = divmod(piece, 6)
                for target in range(64):
                    tr, tf = divmod(target, 8)
                    dr, df = tr - rank, tf - file
                    attacks = (
                        (kind == 0 and dr == (1 if color == 0 else -1) and abs(df) == 1)
                        or (kind == 1 and sorted((abs(dr), abs(df))) == [1, 2])
                        or (kind == 5 and max(abs(dr), abs(df)) == 1)
                    )
                    if attacks:
                        direct[12 + color * 64 + target, square * 12 + piece] = 1
            for dr, df in directions:
                between = []
                tr, tf = rank + dr, file + df
                while 0 <= tr < 8 and 0 <= tf < 8:
                    target = 8 * tr + tf
                    for color in range(2):
                        row = torch.zeros(BOARD_TOKENS)
                        row[square * 12 + color * 6 + (2 if dr and df else 3)] = 1
                        row[square * 12 + color * 6 + 4] = 1
                        for blocker in between:
                            row[blocker * 12 : (blocker + 1) * 12] = -1
                        rays.append(row)
                        destinations.append(12 + color * 64 + target)
                    between.append(target)
                    tr, tf = tr + dr, tf + df
        self.rays = nn.Parameter(torch.stack(rays), requires_grad=False)
        readout = torch.zeros(CONTEXT_DIM, len(rays))
        readout[destinations, torch.arange(len(rays))] = 1
        self.readout = nn.Parameter(readout, requires_grad=False)
        self.direct = nn.Parameter(direct, requires_grad=False)

    def forward(self, totals):
        return nn.functional.linear(totals, self.direct) + nn.functional.linear(
            nn.functional.linear(totals, self.rays).relu(), self.readout
        )


def _prefix_totals(features, ids, open_before=None, totals_before=None):
    valid = ids < MOVE
    if open_before is not None:
        valid = valid & open_before
    valid = (~valid).to(torch.int32).cumsum(dim=-1) == 0
    embedded = features.embedding(ids.clamp(max=MOVE - 1))
    totals = (embedded * valid.unsqueeze(-1)).float().cumsum(dim=-2)
    if totals_before is not None:
        totals = totals + totals_before
    return totals, valid[..., -1:]


class BoardContextAttention(nn.Module):
    """Modulate projected Q/K, preserving the wrapped attention's memory rule."""

    def __init__(self, attention, rank):
        super().__init__()
        self.attention = attention
        self.widths = attention.q.out_features, attention.k.out_features
        self.context_norm = RMSNorm(CONTEXT_DIM)
        self.context_down = nn.Linear(CONTEXT_DIM, rank, bias=False)
        self.context_up = nn.Linear(rank, sum(self.widths), bias=False)
        nn.init.normal_(self.context_up.weight, std=0.001)

    def _condition(self, vectors, context, indices):
        q, k, v = vectors
        scales = self.context_up(
            nn.functional.silu(self.context_down(self.context_norm(context)))
        )[indices]
        q_scale, k_scale = scales.split(self.widths, -1)
        q = q * (1 + q_scale.reshape_as(q).to(q.dtype))
        k = k * (1 + k_scale.reshape_as(k).to(k.dtype))
        return q, k, v

    def forward(self, x, prepared):
        metadata, context, indices = prepared
        attention = self.attention
        if isinstance(attention, RecurrentAttention):
            vectors = self._condition(attention._project(x, metadata), context, indices)
            output, _ = attention._memory(vectors, x, metadata)
            return attention._output(x, output)
        cu, maximum, ends = metadata
        q, k, v = self._condition(attention._project(x), context, indices)
        return attention.o(
            varlen_attention(
                q, k, v, cu, max_seqlen=maximum, sequence_ends=ends
            ).reshape(x.shape)
        )

    def forward_cached(self, x, prepared, cache=None):
        cu, context, indices = prepared
        if len(cu) != 2 or (cache is not None and x.shape[0] != 1):
            raise ValueError(
                "Cached board conditioning requires one document and one new token"
            )
        attention = self.attention
        if isinstance(attention, RecurrentAttention):
            metadata = attention.prepare([0, x.shape[0]], x.device)
            memory, histories = (None, (None, None, None)) if cache is None else cache
            vectors, new_histories = attention._project_cached(x, metadata, histories)
            vectors = self._condition(vectors, context, indices)
            output, state = attention._memory(vectors, x, metadata, memory, True)
            return attention._output(x, output), (state, new_histories)
        q, k, v = self._condition(attention._project(x), context, indices)
        if cache is not None:
            k = torch.cat((cache[0], k))
            v = torch.cat((cache[1], v))
        result = varlen_attention(q, k, v, cu, q.shape[0] == k.shape[0])
        return attention.o(result.reshape(x.shape)), (k, v)


class BoardContextModel(VanillaModel):
    """Condition Q/K scales on a learned sum of observed root-prefix atoms."""

    architecture = "board_context"
    features_cls = LearnedBoardFeatures

    def __init__(self, *, mixer="softmax", softmax_every=0, context_rank=64, **kwargs):
        if mixer not in {"softmax", "gdn", "kda"}:
            raise ValueError("Board context mixer must be softmax, gdn, or kda")
        if not 0 <= softmax_every <= kwargs["num_blocks"]:
            raise ValueError("softmax_every must be zero or a valid layer interval")
        if mixer == "softmax" and softmax_every:
            raise ValueError("softmax_every requires a delta mixer")
        if (
            mixer != "softmax"
            and not softmax_every
            and kwargs["num_kv_heads"] != kwargs["num_heads"]
        ):
            raise ValueError("Pure memory models require num_kv_heads = num_heads")
        if type(context_rank) is not int or context_rank < 1:
            raise ValueError("context_rank must be a positive integer")
        self.mixer, self.softmax_every = mixer, softmax_every
        super().__init__(**kwargs)
        # Build the complete backbone before drawing any conditioning weights.
        for block in self.blocks:
            block.attention = BoardContextAttention(
                block.attention, min(context_rank, kwargs["hidden_size"])
            )
        self.features = self.features_cls()

    def attention_type(self, index):
        if self.mixer == "softmax" or (
            self.softmax_every and (index + 1) % self.softmax_every == 0
        ):
            return CausalSelfAttention
        return {"gdn": GatedDeltaAttention, "kda": KimiDeltaAttention}[self.mixer]

    def _prepare_context(self, input_ids, cu_seqlens):
        cu = torch.as_tensor(cu_seqlens, dtype=torch.long, device="cpu")
        lengths = cu[1:] - cu[:-1]
        offsets = torch.arange(PREFIX_LENGTH)
        gather = (cu[:-1, None] + offsets).clamp(max=input_ids.numel() - 1)
        ids = input_ids[gather.to(input_ids.device)]
        padding = (offsets >= lengths[:, None]).to(input_ids.device)
        ids = ids.masked_fill(padding, MOVE)
        totals, _ = _prefix_totals(self.features, ids)
        context = self.features(totals).flatten(0, 1)
        doc = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
        local = torch.arange(input_ids.numel()) - cu[doc]
        indices = doc * PREFIX_LENGTH + local.clamp(max=PREFIX_LENGTH - 1)
        return context, indices.to(input_ids.device)

    def forward(self, input_ids, cu_seqlens):
        context, indices = self._prepare_context(input_ids, cu_seqlens)
        prepared = {
            kind: (kind.prepare(cu_seqlens, input_ids.device), context, indices)
            for kind in {type(block.attention.attention) for block in self.blocks}
        }
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            metadata = prepared[type(block.attention.attention)]
            if self.training and self.activation_checkpointing:
                x = checkpoint(block, x, metadata, use_reentrant=False)
            else:
                x = block(x, metadata)
        return self.head(self.norm(x))

    def forward_cached(self, input_ids, cu_seqlens, caches=None):
        if caches is None:
            ids = input_ids[:PREFIX_LENGTH]
            totals, open_prefix = _prefix_totals(self.features, ids)
            context = self.features(totals)
            seen = input_ids.numel()
            indices = torch.arange(seen, device=input_ids.device).clamp(
                max=len(ids) - 1
            )
            totals = totals[-1:]
            block_caches = None
        else:
            totals, open_prefix, seen, context, block_caches = caches
            if input_ids.numel() != 1:
                raise ValueError("Incremental board-context decoding accepts one token")
            if seen < PREFIX_LENGTH:
                totals, open_prefix = _prefix_totals(
                    self.features, input_ids, open_prefix, totals
                )
                context = self.features(totals)
            seen += 1
            indices = torch.zeros(1, dtype=torch.long, device=input_ids.device)
        prepared = cu_seqlens, context, indices
        x = self.token_embedding(input_ids)
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = None if block_caches is None else block_caches[i]
            x, cache = block.forward_cached(x, prepared, cache)
            new_caches.append(cache)
        return self.head(self.norm(x)), (
            totals,
            open_prefix,
            seen,
            context[-1:],
            new_caches,
        )


class SymbolicBoardModel(BoardContextModel):
    """Condition attention on piece/attack counts from a fixed root-board circuit."""

    architecture = "symbolic_board"
    features_cls = CompiledBoardFeatures
