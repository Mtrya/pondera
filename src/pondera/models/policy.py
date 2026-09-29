"""Board policies using the lossless input format and composable backbones."""

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from pondera.chess.tokens import encode_boards
from pondera.chess.mapping import IDX_TO_UCI_MOVE, UCI_MOVE_TO_IDX
from pondera.models.backbones import BaseBackbone, DepthBackbone
from pondera.models.board import BoardEmbedding, black_to_move
from pondera.models.config import BlockConfig, CoreConfig


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

        self.register_buffer(
            "reflected_actions",
            torch.tensor(
                [
                    UCI_MOVE_TO_IDX[reflect_move(IDX_TO_UCI_MOVE[i])]
                    for i in range(possible_moves)
                ]
            ),
            persistent=False,
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
        kwargs = {"valid": valid, "capture": capture}
        if isinstance(self.backbone, DepthBackbone):
            kwargs["repetition_limits"] = repetition_limits
        elif repetition_limits is not None:
            raise ValueError("A base policy has no repetition limits")
        output = self.backbone(hidden, **kwargs)
        logits = self.act_proj(output.hidden[:, -2])
        value = self.val_proj(output.hidden[:, -1])
        reflected = black_to_move(ids, positions).to(logits.device)
        logits = torch.where(
            reflected[:, None], logits[:, self.reflected_actions], logits
        )
        return (logits, value, output) if return_backbone else (logits, value)


class DepthPolicyModel(BoardPolicyModel):
    """A board policy with independent repeated cores and the same action interface."""

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
    ):
        groups = [
            CoreConfig(**value) if isinstance(value, dict) else value for value in cores
        ]
        super().__init__(
            1,
            hidden_size,
            intermediate_size,
            num_heads,
            dropout,
            possible_moves,
            architecture,
        )
        self.backbone = DepthBackbone(self.backbone.block_config, groups, cache_policy)
