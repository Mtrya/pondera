"""Board policies using the lossless input format and composable backbones."""

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from pondera.chess.board_tokens import encode_boards
from pondera.chess.mapping import (
    CLAIM_DRAW_FROM,
    IDX_TO_UCI_MOVE,
    MOVE_FROM,
    MOVE_PROMO,
    MOVE_TO,
    REFLECTED_SQUARE,
    UCI_MOVE_TO_IDX,
)
from pondera.models.backbones import BaseBackbone, DepthBackbone
from pondera.models.board import BoardEmbedding, black_to_move
from pondera.models.config import BlockConfig, CoreConfig


class BoardPolicyModel(nn.Module, PyTorchModelHubMixin):
    """Fixed-depth policy used to compare square and compact-piece inputs."""

    def __init__(
        self,
        num_blocks,
        hidden_size,
        intermediate_size,
        num_heads,
        dropout=0.0,
        layout="squares",
        geometry="absolute",
        possible_moves=len(IDX_TO_UCI_MOVE),
        architecture="base",
        encoding="additive",
        orientation="absolute",
        metadata="tokens",
        readout="single",
    ):
        super().__init__()
        if architecture not in {"base", "depth"}:
            raise ValueError("Unsupported board-policy architecture")
        if readout not in {"single", "wide", "factored"}:
            raise ValueError("Unsupported move readout")
        self.layout = layout
        self.readout = readout
        self.move_tokens = {"single": 1, "factored": 2, "wide": 4}[readout]
        self.board_embedding = BoardEmbedding(
            hidden_size, layout, encoding, orientation, geometry, metadata
        )
        self.orientation = orientation
        if orientation == "side_to_move":
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
            if readout == "factored":
                self.register_buffer(
                    "reflected_from",
                    torch.tensor(REFLECTED_SQUARE + (CLAIM_DRAW_FROM,)),
                    persistent=False,
                )
                self.register_buffer(
                    "reflected_to",
                    torch.tensor(REFLECTED_SQUARE),
                    persistent=False,
                )
        self.val_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.val_proj = nn.Linear(hidden_size, 1)
        if readout == "single":
            self.act_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
            self.act_proj = nn.Linear(hidden_size, possible_moves)
        elif readout == "wide":
            self.act_tokens = nn.Parameter(
                torch.randn(1, self.move_tokens, hidden_size) * 0.02
            )
            self.act_head = nn.Sequential(
                nn.Linear(self.move_tokens * hidden_size, self.move_tokens * hidden_size),
                nn.GELU(),
                nn.Linear(self.move_tokens * hidden_size, possible_moves),
            )
        else:
            self.from_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
            self.to_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
            self.from_proj = nn.Linear(hidden_size, CLAIM_DRAW_FROM + 1)
            self.to_proj = nn.Linear(hidden_size, 64)
            self.promo_proj = nn.Linear(hidden_size, 5)
            self.register_buffer(
                "mat_from", torch.tensor(MOVE_FROM), persistent=False
            )
            self.register_buffer(
                "mat_to", torch.tensor(MOVE_TO).clamp_min(0), persistent=False
            )
            self.register_buffer(
                "mat_promo", torch.tensor(MOVE_PROMO).clamp_min(0), persistent=False
            )
            self.register_buffer(
                "has_top",
                (torch.tensor(MOVE_TO) >= 0).float(),
                persistent=False,
            )
        block = BlockConfig(
            hidden_size, intermediate_size, num_heads, dropout, geometry
        )
        self.backbone = BaseBackbone(block, num_blocks)

    def embed(self, ids, positions=None):
        device = self.val_token.device
        ids = ids.to(device)
        if positions is not None:
            positions = positions.to(device)
        board = self.board_embedding(ids, positions)
        batch = ids.shape[0]
        if self.readout == "single":
            readout = self.act_token.expand(batch, -1, -1)
        elif self.readout == "wide":
            readout = self.act_tokens.expand(batch, -1, -1)
        else:
            readout = torch.cat(
                (
                    self.from_token.expand(batch, -1, -1),
                    self.to_token.expand(batch, -1, -1),
                ),
                dim=1,
            )
        count = readout.shape[1] + 1
        hidden = torch.cat(
            (board.hidden, readout, self.val_token.expand(batch, -1, -1)), dim=1
        )
        valid = torch.cat(
            (
                board.valid,
                torch.ones(batch, count, device=device, dtype=torch.bool),
            ),
            dim=1,
        )
        squares = torch.cat(
            (
                board.squares,
                torch.full((batch, count), -1, device=device, dtype=torch.long),
            ),
            dim=1,
        )
        return hidden, valid, squares

    def _factors(self, states, ids, positions):
        """Absolute-orientation from/to/promotion logits from the readout states."""
        from_logits = self.from_proj(states[:, 0])
        to_logits = self.to_proj(states[:, 1])
        promo_logits = self.promo_proj(states[:, 1])
        if self.orientation == "side_to_move":
            reflected = black_to_move(ids, positions).to(from_logits.device)
            from_logits = torch.where(
                reflected[:, None], from_logits[:, self.reflected_from], from_logits
            )
            to_logits = torch.where(
                reflected[:, None], to_logits[:, self.reflected_to], to_logits
            )
        return from_logits, to_logits, promo_logits

    def move_log_probs(self, factors):
        """Materialize per-move log-probabilities over the full move vocabulary."""
        from_logits, to_logits, promo_logits = (f.float() for f in factors)
        joint = from_logits.log_softmax(-1)[:, self.mat_from]
        return joint + (
            to_logits.log_softmax(-1)[:, self.mat_to]
            + promo_logits.log_softmax(-1)[:, self.mat_promo]
        ) * self.has_top

    def forward(
        self,
        fen=None,
        repetitions=None,
        *,
        ids=None,
        positions=None,
        return_backbone=False,
        return_factors=False,
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
        hidden, valid, squares = self.embed(ids, positions)
        kwargs = {"valid": valid, "squares": squares, "capture": capture}
        if isinstance(self.backbone, DepthBackbone):
            kwargs["repetition_limits"] = repetition_limits
        elif repetition_limits is not None:
            raise ValueError("A base policy has no repetition limits")
        output = self.backbone(hidden, **kwargs)
        states = output.hidden[:, -1 - self.move_tokens : -1]
        value = self.val_proj(output.hidden[:, -1])
        if self.readout == "factored":
            factors = self._factors(states, ids, positions)
            predictions = (
                (factors, value) if return_factors else (self.move_log_probs(factors), value)
            )
        else:
            if self.readout == "single":
                logits = self.act_proj(states[:, 0])
            else:
                logits = self.act_head(states.reshape(states.shape[0], -1))
            if self.orientation == "side_to_move":
                reflected = black_to_move(ids, positions).to(logits.device)
                logits = torch.where(
                    reflected[:, None], logits[:, self.reflected_actions], logits
                )
            predictions = (logits, value)
        return (*predictions, output) if return_backbone else predictions


class DepthPolicyModel(BoardPolicyModel):
    """A board policy with independent repeated cores and the same action interface."""

    def __init__(
        self,
        cores,
        hidden_size,
        intermediate_size,
        num_heads,
        dropout=0.0,
        layout="squares",
        geometry="absolute",
        possible_moves=len(IDX_TO_UCI_MOVE),
        cache_policy="matched_fallback",
        architecture="depth",
        encoding="additive",
        orientation="absolute",
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
            layout,
            geometry,
            possible_moves,
            architecture,
            encoding,
            orientation,
        )
        self.backbone = DepthBackbone(self.backbone.block_config, groups, cache_policy)
