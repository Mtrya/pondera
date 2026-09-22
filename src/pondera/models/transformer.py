from typing import List

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from pondera.chess.mapping import (
    IDX_TO_UCI_MOVE,
    MAX_FULLMOVES,
    MAX_HALFMOVES,
)
from pondera.chess.tokenize import encode_fens


# --- Tokenizer --- #
class FENTokenizer(nn.Module):
    """Convert FEN (and repetitions) to a sequence of tokens"""

    def __init__(self, hidden_size, dtype):
        super().__init__()

        self.side_embed = nn.Embedding(
            2, hidden_size, dtype=dtype
        )  # black/white embedding

        self.castling_embed_k = nn.Parameter(
            torch.randn(1, 1, hidden_size, dtype=dtype)
        )
        self.castling_embed_q = nn.Parameter(
            torch.randn(1, 1, hidden_size, dtype=dtype)
        )
        self.castling_embed_K = nn.Parameter(
            torch.randn(1, 1, hidden_size, dtype=dtype)
        )
        self.castling_embed_Q = nn.Parameter(
            torch.randn(1, 1, hidden_size, dtype=dtype)
        )
        self.no_castling_embed = nn.Parameter(
            torch.randn(1, 1, hidden_size, dtype=dtype)
        )

        self.piece_embed = nn.Embedding(
            13, hidden_size, dtype=dtype
        )  # 6 for white pieces, 6 for black pieces, 1 for empty

        self.no_en_passant_embed = nn.Parameter(
            torch.randn(1, 1, hidden_size, dtype=dtype)
        )  # use positional embed for the target square, or a special one for '-'

        self.half_move_embed = nn.Embedding(MAX_HALFMOVES, hidden_size, dtype=dtype)

        self.full_move_embed = nn.Embedding(MAX_FULLMOVES, hidden_size, dtype=dtype)

        self.repetition_embed = nn.Embedding(3, hidden_size, dtype=dtype)

        self.pos_embed = nn.Embedding(
            64, hidden_size, dtype=dtype
        )  # positional embedding

    def forward(self, fen_list: List[str], repetitions: torch.Tensor) -> torch.Tensor:
        """Encode FEN input using the same representation as prepared datasets."""
        ids = encode_fens(fen_list, repetitions.detach().cpu().tolist())
        return self.forward_ids(torch.from_numpy(ids))

    def forward_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Pre-encoded token ids -> embeddings. Mirrors forward() on parsed FENs.

        Args:
            ids: (B, 73) integer tensor with the layout from pondera.chess.tokenize

        Returns:
            (B, 73, hidden_size)
        """
        device = self.side_embed.weight.device
        ids = ids.to(device=device, dtype=torch.long)
        bs = ids.shape[0]

        square_indices = torch.arange(64, device=device)
        board_tokens = self.piece_embed(ids[:, :64]) + self.pos_embed(
            square_indices
        )  # (B,64,D)

        side_token = self.side_embed(ids[:, 64]).unsqueeze(1)  # (B,1,D)

        castling_params = torch.cat(
            [
                self.castling_embed_K,
                self.castling_embed_Q,
                self.castling_embed_k,
                self.castling_embed_q,
            ],
            dim=1,
        ).view(4, -1)  # (4,D)
        castling_present = ids[:, 65:69].unsqueeze(-1).bool()  # (B,4,1)
        castling_tokens = torch.where(
            castling_present,
            castling_params.unsqueeze(0),
            self.no_castling_embed.view(1, 1, -1),
        )  # (B,4,D)

        ep_ids = ids[:, 69]
        ep_square_tokens = self.pos_embed(ep_ids.clamp(max=63))  # (B,D)
        ep_token = torch.where(
            (ep_ids < 64).unsqueeze(-1),
            ep_square_tokens,
            self.no_en_passant_embed.view(1, -1).expand(bs, -1),
        ).unsqueeze(1)  # (B,1,D)

        half_move_token = self.half_move_embed(ids[:, 70]).unsqueeze(1)  # (B,1,D)
        full_move_token = self.full_move_embed(ids[:, 71]).unsqueeze(1)  # (B,1,D)
        repetition_token = self.repetition_embed(ids[:, 72]).unsqueeze(1)  # (B,1,D)

        return torch.cat(
            [
                board_tokens,
                side_token,
                castling_tokens,
                ep_token,
                half_move_token,
                full_move_token,
                repetition_token,
            ],
            dim=1,
        )  # (B,73,D)


# --- Helper Modules --- #
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
        x = self.up_proj(x) * self.dropout(nn.functional.silu(self.gate_proj(x)))
        return self.down_proj(x)


class TransformerEncoderLayer(nn.Module):
    """Custom transformer encoder layer with RMSNorm and SwiGLUFFN"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        batch_first: bool = True,
        norm_first: bool = False,
        dtype=None,
    ):
        super().__init__()
        self.norm_first = norm_first

        self.norm1 = nn.RMSNorm(d_model, dtype=dtype)
        self.dropout_sa = nn.Dropout(dropout)
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=False,
            batch_first=batch_first,
            dtype=dtype,
        )

        self.norm2 = nn.RMSNorm(d_model, dtype=dtype)
        self.dropout_ff = nn.Dropout(dropout)
        self.mlp = SwiGLUFFN(
            d_model,
            dim_feedforward,
            dropout=dropout,
            bias_up=False,
            bias_gate=False,
            bias_down=True,
            dtype=dtype,
        )

    def forward(self, x, return_attention=False):
        if self.norm_first:
            if return_attention:
                x_norm = self.norm1(x)
                attn_output, attn_weights = self._sa_block(
                    x_norm, return_attention=True
                )
                x = x + attn_output
                x = x + self._ff_block(self.norm2(x))
                return x, attn_weights
            else:
                x = x + self._sa_block(self.norm1(x))
                x = x + self._ff_block(self.norm2(x))
                return x
        else:
            if return_attention:
                attn_output, attn_weights = self._sa_block(x, return_attention=True)
                x = self.norm1(x + attn_output)
                x = self.norm2(x + self._ff_block(x))
                return x, attn_weights
            else:
                x = self.norm1(x + self._sa_block(x))
                x = self.norm2(x + self._ff_block(x))
                return x

    def _sa_block(self, x, return_attention=False):
        if return_attention:
            attn_output, attn_weights = self.self_attn(
                x, x, x, need_weights=True, average_attn_weights=False
            )
            return self.dropout_sa(attn_output), attn_weights
        else:
            x = self.self_attn(x, x, x, need_weights=False)[0]
            return self.dropout_sa(x)

    def _ff_block(self, x):
        x = self.mlp(x)
        return self.dropout_ff(x)


# --- Model Arch --- #
class PonderaModel(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        num_blocks,
        hidden_size,
        intermediate_size,
        num_heads,
        dropout: float = 0.00,
        possible_moves: int = len(IDX_TO_UCI_MOVE),  # 1969 structurally valid moves
        dtype=None,
    ):
        super().__init__()
        self.fen_tokenizer = FENTokenizer(hidden_size, dtype=dtype)

        self.act_token = nn.Parameter(
            torch.randn((1, 1, hidden_size), dtype=dtype) * 0.02
        )
        self.val_token = nn.Parameter(
            torch.randn((1, 1, hidden_size), dtype=dtype) * 0.02
        )

        self.act_proj = nn.Linear(hidden_size, possible_moves, dtype=dtype)
        self.val_proj = nn.Linear(hidden_size, 1, dtype=dtype)

        self.blocks = nn.ModuleList(
            TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=intermediate_size,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                dtype=dtype,
            )
            for _ in range(num_blocks)
        )
        self.dtype = dtype
        self.possible_moves = possible_moves

        self.final_norm = nn.RMSNorm(hidden_size)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                if hasattr(m, "weight"):
                    nn.init.constant_(m.weight, 1.0)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.weight, 0.0)
            elif isinstance(m, nn.RMSNorm):
                if hasattr(m, "weight"):
                    nn.init.constant_(m.weight, 1.0)

        tokenizer_params = dict(self.fen_tokenizer.named_parameters())

        params_to_init = [
            self.act_token,
            self.val_token,
            tokenizer_params.get("castling_embed_k"),
            tokenizer_params.get("castling_embed_q"),
            tokenizer_params.get("castling_embed_K"),
            tokenizer_params.get("castling_embed_Q"),
            tokenizer_params.get("no_castling_embed"),
            tokenizer_params.get("no_en_passant_embed"),
        ]

        for param in params_to_init:
            if param is not None and param.requires_grad:
                nn.init.normal_(param, std=0.02)

    def forward(
        self,
        fen: List[str] = None,
        repetitions: torch.Tensor = None,
        return_attention: bool = False,
        ids: torch.Tensor = None,
    ) -> torch.Tensor:
        if ids is not None:
            x = self.fen_tokenizer.forward_ids(ids)  # (B,73,D)
        else:
            x = self.fen_tokenizer(fen, repetitions)  # (B,73,D)
        bs = x.shape[0]
        x = torch.cat(
            [x, self.act_token.expand(bs, -1, -1), self.val_token.expand(bs, -1, -1)],
            dim=1,
        )  # (B,75,D)

        attention_maps = [] if return_attention else None

        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attention_maps.append(attn)
            else:
                x = block(x)

        x = self.final_norm(x)

        act = x[:, -2, :]
        val = x[:, -1, :]
        act_logits = self.act_proj(act)  # (B,1969)
        val = self.val_proj(val)  # (B,1)

        if return_attention:
            return act_logits, val.squeeze(1), attention_maps
        else:
            return act_logits, val.squeeze(1)
