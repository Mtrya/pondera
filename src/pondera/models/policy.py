"""Board policies using the lossless input format and composable backbones."""

from dataclasses import dataclass

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from pondera.chess.tokens import (
    CHESS_VOCAB_SIZE,
    MOVE,
    VALUE,
    encode_boards,
)
from pondera.chess.mapping import IDX_TO_UCI_MOVE, UCI_MOVE_TO_IDX
from pondera.models.backbones import BaseBackbone, DepthBackbone
from pondera.models.board import BoardEmbedding, black_to_move
from pondera.models.config import BlockConfig, CoreConfig


def _reflection_table():
    """Action-permutation undoing the black-to-move rank reflection."""

    def reflect_move(move):
        if move == "<claim_draw>":
            return move
        return (
            move[0]
            + str(9 - int(move[1]))
            + move[2]
            + str(9 - int(move[3]))
            + move[4:]
        )

    return torch.tensor(
        [
            UCI_MOVE_TO_IDX[reflect_move(IDX_TO_UCI_MOVE[i])]
            for i in range(len(IDX_TO_UCI_MOVE))
        ]
    )


class BoardPolicyModel(nn.Module, PyTorchModelHubMixin):
    """Fixed-depth policy with a single-token move readout over the full vocabulary."""

    def __init__(
        self,
        num_blocks,
        hidden_size,
        intermediate_size,
        num_heads,
        dropout=0.0,
        possible_moves=len(IDX_TO_UCI_MOVE),
        architecture="base",
    ):
        super().__init__()
        if architecture not in {"base", "depth"}:
            raise ValueError("Unsupported board-policy architecture")
        self.board_embedding = BoardEmbedding(hidden_size)
        if possible_moves != len(IDX_TO_UCI_MOVE):
            raise ValueError(
                "Side-to-move orientation requires the full action vocabulary"
            )
        self.register_buffer(
            "reflected_actions", _reflection_table(), persistent=False
        )
        self.val_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.val_proj = nn.Linear(hidden_size, 1)
        self.act_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.act_proj = nn.Linear(hidden_size, possible_moves)
        block = BlockConfig(hidden_size, intermediate_size, num_heads, dropout)
        self.backbone = BaseBackbone(block, num_blocks)

    def embed(self, ids, positions=None):
        device = self.val_token.device
        ids = ids.to(device)
        if positions is not None:
            positions = positions.to(device)
        board = self.board_embedding(ids, positions)
        batch = ids.shape[0]
        hidden = torch.cat(
            (
                board.hidden,
                self.act_token.expand(batch, -1, -1),
                self.val_token.expand(batch, -1, -1),
            ),
            dim=1,
        )
        valid = torch.cat(
            (board.valid, torch.ones(batch, 2, device=device, dtype=torch.bool)),
            dim=1,
        )
        return hidden, valid

    def forward(
        self,
        fen=None,
        repetitions=None,
        *,
        ids=None,
        positions=None,
        return_backbone=False,
        capture=False,
        repetition_limits=None,
    ):
        if ids is None:
            reps = (
                repetitions.tolist()
                if isinstance(repetitions, torch.Tensor)
                else repetitions
            )
            ids = torch.from_numpy(encode_boards(fen, reps))
        hidden, valid = self.embed(ids, positions)
        if repetition_limits is not None:
            raise ValueError("A base policy has no repetition limits")
        output = self.backbone(hidden, valid=valid, capture=capture)
        logits = self.act_proj(output.hidden[:, -2])
        value = self.val_proj(output.hidden[:, -1])
        reflected = black_to_move(ids, positions).to(logits.device)
        logits = torch.where(
            reflected[:, None], logits[:, self.reflected_actions], logits
        )
        return (logits, value, output) if return_backbone else (logits, value)


@dataclass
class ExitPrediction:
    """One supervised core exit: move and value logits plus the active mask."""

    move_logits: torch.Tensor
    value_logits: torch.Tensor
    active: torch.Tensor


class DepthPolicyModel(nn.Module, PyTorchModelHubMixin):
    """Recurrent-depth policy; one shared vocabulary head read at every core exit.

    The head spans the unified token vocabulary. The action readout consumes its
    MOVE segment, the value readout its VALUE/MATE segment; both sit at the same
    trailing readout positions as the fixed-depth policy. Exits are the core
    repetition boundaries in execution order; the final answer is the last exit.
    With exit_adapter enabled, each exit additionally passes through its own
    residual single-layer adapter before the shared head.
    """

    def __init__(
        self,
        cores,
        hidden_size,
        intermediate_size,
        num_heads,
        dropout=0.0,
        possible_moves=len(IDX_TO_UCI_MOVE),
        cache_policy="matched_fallback",
        architecture="depth",
        exit_adapter=False,
    ):
        super().__init__()
        if architecture != "depth":
            raise ValueError("Unsupported board-policy architecture")
        if possible_moves != len(IDX_TO_UCI_MOVE):
            raise ValueError(
                "Side-to-move orientation requires the full action vocabulary"
            )
        self.board_embedding = BoardEmbedding(hidden_size)
        self.register_buffer(
            "reflected_actions", _reflection_table(), persistent=False
        )
        self.act_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.val_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.head = nn.Linear(hidden_size, CHESS_VOCAB_SIZE)
        groups = [
            CoreConfig(**value) if isinstance(value, dict) else value for value in cores
        ]
        block = BlockConfig(hidden_size, intermediate_size, num_heads, dropout)
        self.backbone = DepthBackbone(block, groups, cache_policy)
        num_exits = sum(core.repetition.max_repetitions for core in groups)
        self.exit_adapters = (
            nn.ModuleList(
                nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU())
                for _ in range(num_exits)
            )
            if exit_adapter
            else None
        )

    def embed(self, ids, positions=None):
        device = self.act_token.device
        ids = ids.to(device)
        if positions is not None:
            positions = positions.to(device)
        board = self.board_embedding(ids, positions)
        batch = ids.shape[0]
        hidden = torch.cat(
            (
                board.hidden,
                self.act_token.expand(batch, -1, -1),
                self.val_token.expand(batch, -1, -1),
            ),
            dim=1,
        )
        valid = torch.cat(
            (board.valid, torch.ones(batch, 2, device=device, dtype=torch.bool)),
            dim=1,
        )
        return hidden, valid

    def _predict(self, state, reflected):
        logits = self.head(state)
        move = logits[..., 0, MOVE : MOVE + len(IDX_TO_UCI_MOVE)]
        value = logits[..., 1, VALUE:]
        move = torch.where(
            reflected[:, None], move[:, self.reflected_actions], move
        )
        return move, value

    def forward(
        self,
        fen=None,
        repetitions=None,
        *,
        ids=None,
        positions=None,
        return_backbone=False,
        capture=False,
        repetition_limits=None,
    ):
        if ids is None:
            reps = (
                repetitions.tolist()
                if isinstance(repetitions, torch.Tensor)
                else repetitions
            )
            ids = torch.from_numpy(encode_boards(fen, reps))
        hidden, valid = self.embed(ids, positions)
        output = self.backbone(
            hidden,
            valid=valid,
            capture=capture,
            repetition_limits=repetition_limits,
        )
        reflected = black_to_move(ids, positions).to(output.hidden.device)
        state = output.hidden[:, -2:]
        if self.exit_adapters is not None:
            state = state + self.exit_adapters[-1](state)
        move, value = self._predict(state, reflected)
        return (move, value, output) if return_backbone else (move, value)

    def forward_exits(self, *, ids, positions=None, repetition_limits=None):
        """Every core-exit prediction in execution order, for multi-exit training."""
        hidden, valid = self.embed(ids, positions)
        output = self.backbone(
            hidden,
            valid=valid,
            capture=True,
            repetition_limits=repetition_limits,
        )
        reflected = black_to_move(ids, positions).to(output.hidden.device)
        last_layers = tuple(core.config.layers - 1 for core in self.backbone.cores)
        exits = []
        for readout in output.readouts:
            if readout.layer != last_layers[readout.core]:
                continue
            normed = self.backbone.final_norm(readout.hidden)
            state = normed[:, -2:]
            if self.exit_adapters is not None:
                state = state + self.exit_adapters[len(exits)](state)
            move, value = self._predict(state, reflected)
            exits.append(ExitPrediction(move, value, readout.active[:, -2]))
        return exits, output
