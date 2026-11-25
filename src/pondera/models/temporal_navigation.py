"""Content-dependent reads from two causal positions after BACK or RESET."""

from copy import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from pondera.models.navigation import (
    BranchEntries,
    navigation_sources,
    navigation_table,
)
from pondera.models.token_value import KimiDeltaTokenValueModel
from pondera.models.token_value_attn_res import TokenValueBlockAttnResModel


@dataclass(frozen=True)
class NavigationCache:
    position: int
    cursor: BranchEntries
    hidden: dict[int, torch.Tensor]


class NavigationRead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(width))
        self.score_scale = width**-0.5
        self.table = ()

    def prepare(self, tokens, boundaries):
        sources = navigation_sources(tokens, boundaries, self.table)
        receivers = np.flatnonzero(sources[:, 0] >= 0)
        return receivers, sources[receivers]

    def _mix(self, current, sources):
        states = torch.cat((current[:, None], sources), dim=1).float()
        keys = states * torch.rsqrt(states.square().mean(dim=-1, keepdim=True) + 1e-6)
        query = keys[:, :1] * self.query.float() * self.score_scale
        scores = (keys * query).sum(dim=-1)
        weights = (3 * scores.softmax(dim=-1)[:, 1:] - 1).unsqueeze(-1)
        return ((states[:, 1:] - states[:, :1]) * weights).sum(dim=1).to(current.dtype)

    def _add(self, hidden, bank, receivers, sources):
        receiver_indices = torch.as_tensor(
            receivers, dtype=torch.long, device=hidden.device
        )
        sender_indices = torch.as_tensor(
            sources, dtype=torch.long, device=hidden.device
        )
        current = hidden.index_select(0, receiver_indices)
        previous = bank.index_select(0, sender_indices.flatten()).unflatten(0, (-1, 2))
        return hidden.index_add(0, receiver_indices, self._mix(current, previous))

    def forward(self, hidden, prepared):
        receivers, sources = prepared
        if not len(receivers):
            return hidden + self.query.sum().to(hidden.dtype) * 0
        return self._add(hidden, hidden, receivers, sources)

    def forward_cached(self, hidden, tokens, cache=None):
        if cache is None:
            cursor, start, prior = BranchEntries(), 0, {}
        else:
            cursor, start, prior = copy(cache.cursor), cache.position, cache.hidden
            cursor.path = cursor.path.copy()
        if len(hidden) != len(tokens):
            raise ValueError("Expected one layer output per input token")
        references = [
            cursor.consume(start + i, self.table[token])
            for i, token in enumerate(tokens)
        ]
        receivers = [i for i, (branch, _) in enumerate(references) if branch >= 0]
        if not receivers:
            output = hidden
        elif len(hidden) == 1:
            earlier = torch.stack(
                [prior[source] for source in references[0]]
            ).unsqueeze(0)
            output = hidden + self._mix(hidden, earlier)
        else:
            bank = (
                torch.cat((hidden, torch.stack(tuple(prior.values()))))
                if prior
                else hidden
            )
            previous = {source: len(hidden) + i for i, source in enumerate(prior)}
            senders = [
                [
                    source - start if source >= start else previous[source]
                    for source in references[i]
                ]
                for i in receivers
            ]
            output = self._add(hidden, bank, receivers, senders)
        needed = (
            set()
            if cursor.final
            else {cursor.root, cursor.latest_move, *cursor.path} - {-1}
        )
        saved = {
            index: prior[index] if index < start else hidden[index - start].clone()
            for index in needed
        }
        return output, NavigationCache(start + len(tokens), cursor, saved)


class TokenValueNavigationModel(KimiDeltaTokenValueModel):
    architecture = "kda_token_value_navigation"

    def __init__(self, *, hidden_size, **settings):
        super().__init__(hidden_size=hidden_size, **settings)
        if len(self.blocks) < 4:
            raise ValueError("Chess-path reads require at least four blocks")
        self.navigation = NavigationRead(hidden_size)

    def bind_tokenizer(self, dictionary):
        self.navigation.table = navigation_table(dictionary)
        if len(self.navigation.table) != self.vocab_size:
            raise ValueError("Navigation dictionary must match the model vocabulary")

    def forward(self, input_ids, cu_seqlens, *, navigation_reads=None):
        if navigation_reads is None:
            navigation_reads = self.navigation.prepare(
                input_ids.detach().cpu().numpy(), cu_seqlens
            )

        def read_after_fourth_block(index, residual):
            if index != 3:
                return residual
            if self.training and self.activation_checkpointing:
                return checkpoint(
                    self.navigation, residual, navigation_reads, use_reentrant=False
                )
            return self.navigation(residual, navigation_reads)

        return super().forward(
            input_ids, cu_seqlens, after_block=read_after_fourth_block
        )

    def forward_cached(self, input_ids, cu_seqlens, caches=None, *, host_tokens=None):
        block_caches, navigation_cache = (None, None) if caches is None else caches
        if host_tokens is None:
            host_tokens = input_ids.detach().cpu().tolist()

        def read_after_fourth_block(index, residual):
            nonlocal navigation_cache
            if index == 3:
                residual, navigation_cache = self.navigation.forward_cached(
                    residual, host_tokens, navigation_cache
                )
            return residual

        logits, block_caches = super().forward_cached(
            input_ids, cu_seqlens, block_caches, after_block=read_after_fourth_block
        )
        return logits, (block_caches, navigation_cache)


class TokenValueBlockAttnResNavigationModel(
    TokenValueNavigationModel, TokenValueBlockAttnResModel
):
    architecture = "kda_token_value_block_attn_res_navigation"
