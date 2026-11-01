"""Causal policy over board atoms and trace tokens."""

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from pondera.chess.vocabulary import (
    CHESS_TOKENIZER_VERSION,
    CHESS_VOCAB_SIZE,
    PAD,
)
from pondera.models.backbones import Backbone
from pondera.models.blocks import sinusoidal_positions
from pondera.models.board import black_to_move, reflect_board_tokens
from pondera.models.config import BlockConfig


def trace_target_positions(board_lengths, trace_width, board_width):
    index = torch.arange(trace_width, device=board_lengths.device)
    return torch.where(
        index[None, :] == 0,
        (board_lengths - 1)[:, None],
        board_width + index[None, :] - 1,
    )


class PolicyModel(nn.Module, PyTorchModelHubMixin):
    """Bidirectional board prefix followed by a causally attended trace."""

    def __init__(
        self,
        num_blocks,
        hidden_size,
        intermediate_size,
        num_heads,
        dropout=0.0,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(
            CHESS_VOCAB_SIZE, hidden_size, padding_idx=PAD
        )
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[PAD].zero_()
        self.head = nn.Linear(hidden_size, CHESS_VOCAB_SIZE)
        block = BlockConfig(hidden_size, intermediate_size, num_heads, dropout)
        self.backbone = Backbone(block, num_blocks)

    def _prefix_embeddings(self, ids):
        ids = reflect_board_tokens(ids, black_to_move(ids))
        ids = ids.sort(dim=1).values
        valid = ids != PAD
        hidden = self.token_embedding(ids.long())
        return hidden.masked_fill(~valid.unsqueeze(-1), 0), valid

    def _embed(self, ids, trace_ids=None, trace_positions=None):
        hidden, valid = self._prefix_embeddings(ids)
        board_width = ids.shape[1]
        if trace_ids is not None:
            if trace_positions is None:
                trace_positions = torch.arange(
                    trace_ids.shape[1], device=trace_ids.device
                )
            trace = self.token_embedding(trace_ids.long()) + sinusoidal_positions(
                trace_positions, hidden.shape[-1], hidden.dtype
            )
            hidden = torch.cat((hidden, trace), dim=1)
            valid = torch.cat((valid, trace_ids != PAD), dim=1)
        output = self.backbone(hidden, board_width, valid=valid)
        return hidden, valid, board_width, output

    def forward(self, ids=None, trace_ids=None, *, trace_positions=None):
        if ids is None:
            raise ValueError("The policy consumes encoded board tokens")
        *_, output = self._embed(ids, trace_ids, trace_positions)
        return self.head(output.hidden)

    def trace_logits(self, ids, trace_ids=None):
        if trace_ids is None:
            raise ValueError("Trace supervision needs trace tokens")
        _, valid, board_width, output = self._embed(ids, trace_ids)
        target = trace_target_positions(
            valid[:, :board_width].sum(1), trace_ids.shape[1], board_width
        )
        selected = output.hidden.gather(
            1, target.unsqueeze(-1).expand(-1, -1, output.hidden.shape[-1])
        )
        return self.head(selected)

    @torch.no_grad()
    def prefill(self, ids):
        if self.training:
            raise ValueError("Generation requires evaluation mode")
        hidden, valid = self._prefix_embeddings(ids)
        output = self.backbone.prefill(hidden, valid=valid)
        last = valid.sum(1) - 1
        selected = output.hidden.gather(
            1, last[:, None, None].expand(-1, 1, output.hidden.shape[-1])
        )
        return self.head(selected.squeeze(1)), output.cache

    @torch.no_grad()
    def decode(self, tokens, cache):
        if self.training:
            raise ValueError("Generation requires evaluation mode")
        positions = torch.arange(tokens.shape[1], device=tokens.device) + (
            cache.length - cache.prefix_length
        )
        hidden = self.token_embedding(tokens.long()) + sinusoidal_positions(
            positions,
            self.token_embedding.embedding_dim,
            self.token_embedding.weight.dtype,
        )
        output = self.backbone.decode(hidden, cache, valid=tokens != PAD)
        return self.head(output.hidden[:, -1]), output.cache


def create_model(config):
    arguments = dict(config)
    tokenizer_version = arguments.pop("tokenizer_version", CHESS_TOKENIZER_VERSION)
    if tokenizer_version != CHESS_TOKENIZER_VERSION:
        raise ValueError("Only the lossless board input format is supported")
    return PolicyModel(**arguments)
