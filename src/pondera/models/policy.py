"""Board policies using the lossless input format and composable backbones."""

from dataclasses import dataclass

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from pondera.chess.tokens import (
    CHESS_VOCAB_SIZE,
    LINE,
    MOVE,
    PAD,
    SNAPSHOT,
    SNAPSHOT_PIECES,
    VALUE,
    encode_boards,
)
from pondera.chess.mapping import EMPTY_SQ_IDX, IDX_TO_UCI_MOVE
from pondera.chess.symmetry import move_permutations
from pondera.models.backbones import BaseBackbone, DepthBackbone, SequenceBackbone
from pondera.models.blocks import sinusoidal_positions
from pondera.models.board import BoardEmbedding, black_to_move
from pondera.models.config import BlockConfig, CoreConfig


def _reflection_table():
    """Action-permutation undoing the black-to-move rank reflection."""
    return torch.tensor(move_permutations()[1])


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
    ):
        if ids is None:
            reps = (
                repetitions.tolist()
                if isinstance(repetitions, torch.Tensor)
                else repetitions
            )
            ids = torch.from_numpy(encode_boards(fen, reps))
        hidden, valid = self.embed(ids, positions)
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
    board_logits: torch.Tensor | None = None
    active_board: torch.Tensor | None = None


class DepthPolicyModel(nn.Module, PyTorchModelHubMixin):
    """Recurrent-depth policy; one shared vocabulary head read at every core exit.

    The head spans the unified token vocabulary. The action readout consumes its
    MOVE segment, the value readout its VALUE/MATE segment; both sit at the same
    trailing readout positions as the fixed-depth policy. Exits are the core
    repetition boundaries in execution order; the final answer is the last exit.
    With board_aux, a shared 13-class head additionally reads every sequence
    position at each exit, predicting the square contents after replaying the
    exit's teacher move (a training-time auxiliary; the fast path ignores it).
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
        board_aux=False,
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
        self.board_aux_head = (
            nn.Linear(hidden_size, EMPTY_SQ_IDX + 1) if board_aux else None
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
        value = logits[..., 1, VALUE:LINE]
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
    ):
        if ids is None:
            reps = (
                repetitions.tolist()
                if isinstance(repetitions, torch.Tensor)
                else repetitions
            )
            ids = torch.from_numpy(encode_boards(fen, reps))
        hidden, valid = self.embed(ids, positions)
        output = self.backbone(hidden, valid=valid, capture=capture)
        reflected = black_to_move(ids, positions).to(output.hidden.device)
        move, value = self._predict(output.hidden[:, -2:], reflected)
        return (move, value, output) if return_backbone else (move, value)

    def forward_exits(self, *, ids, positions=None):
        """Every core-exit prediction in execution order, for multi-exit training."""
        hidden, valid = self.embed(ids, positions)
        output = self.backbone(hidden, valid=valid, capture=True)
        reflected = black_to_move(ids, positions).to(output.hidden.device)
        last_layers = tuple(core.config.layers - 1 for core in self.backbone.cores)
        exits = []
        for readout in output.readouts:
            if readout.layer != last_layers[readout.core]:
                continue
            normed = self.backbone.final_norm(readout.hidden)
            state = normed[:, -2:]
            move, value = self._predict(state, reflected)
            board_logits = (
                self.board_aux_head(normed) if self.board_aux_head is not None else None
            )
            exits.append(
                ExitPrediction(
                    move,
                    value,
                    readout.active[:, -2],
                    board_logits,
                    readout.active[:, :-2] if board_logits is not None else None,
                )
            )
        return exits, output


def sequence_target_positions(board_lengths, trace_width, board_width):
    """Absolute positions whose hidden state predicts each trace target token.

    The first trace token is predicted from the last valid packed board
    position; every later trace token is predicted from the preceding trace
    position. Padded trace slots get in-range indices and are ignored through
    the loss mask.
    """
    index = torch.arange(trace_width, device=board_lengths.device)
    return torch.where(
        index[None, :] == 0,
        (board_lengths - 1)[:, None],
        board_width + index[None, :] - 1,
    )


class SequencePolicyModel(nn.Module, PyTorchModelHubMixin):
    """Board prefix with bidirectional attention followed by a causal trace.

    The board prefix packs to its batch-maximum width and is embedded by
    BoardEmbedding; padded board slots stay invisible keys, so no sample attends
    to another sample's padding. Trace tokens carry their own embedding plus
    sinusoidal positions, except snapshot tokens, which share the board's joint
    (square, piece) table: reading a board and writing a snapshot use one set of
    parameters. The head spans the unified vocabulary and reads the whole
    sequence; supervision follows the shifted trace contract in
    sequence_target_positions. Generation prefills the board, then decodes one
    token at a time from the cached prefix.
    """

    def __init__(
        self,
        num_blocks,
        hidden_size,
        intermediate_size,
        num_heads,
        dropout=0.0,
        architecture="sequence",
    ):
        super().__init__()
        if architecture != "sequence":
            raise ValueError("Unsupported board-policy architecture")
        self.board_embedding = BoardEmbedding(hidden_size)
        # Rows below the snapshot block; snapshot ids read the board's joint
        # table through _trace_embeddings instead of a second copy of it.
        self.trace_embedding = nn.Embedding(SNAPSHOT, hidden_size, padding_idx=PAD)
        nn.init.normal_(self.trace_embedding.weight, std=0.02)
        with torch.no_grad():
            self.trace_embedding.weight[PAD].zero_()
        self.head = nn.Linear(hidden_size, CHESS_VOCAB_SIZE)
        block = BlockConfig(hidden_size, intermediate_size, num_heads, dropout)
        self.backbone = SequenceBackbone(block, num_blocks)

    def _trace_embeddings(self, trace_ids, trace_positions, dtype):
        """Trace embeddings, with snapshot rows tied to the board square table.

        Snapshot token SNAPSHOT + square * 12 + piece reads row
        square * 13 + piece of board_embedding.squares, so the same parameters
        embed a piece on a square whether the model reads a board or writes a
        snapshot, and snapshot supervision updates that table.
        """
        if trace_positions is None:
            trace_positions = torch.arange(trace_ids.shape[1], device=trace_ids.device)
        ids = trace_ids.long()
        snapshot = ids >= SNAPSHOT
        offset = ids - SNAPSHOT
        square, piece = offset // SNAPSHOT_PIECES, offset % SNAPSHOT_PIECES
        # The board table keeps one extra row per square for the empty square.
        stride = SNAPSHOT_PIECES + 1
        shared = self.board_embedding.squares(
            (square * stride + piece).clamp(0, stride * 64 - 1)
        )
        base = self.trace_embedding(ids.clamp(max=SNAPSHOT - 1))
        content = torch.where(snapshot.unsqueeze(-1), shared, base)
        return content + sinusoidal_positions(
            trace_positions, self.trace_embedding.embedding_dim, dtype
        )

    def sequence(self, ids, positions=None, trace_ids=None, trace_positions=None):
        """Backbone output over the packed board prefix and the trace suffix."""
        board = self.board_embedding(ids, positions)
        hidden, valid = board.hidden, board.valid
        if trace_ids is not None:
            trace = self._trace_embeddings(
                trace_ids, trace_positions, board.hidden.dtype
            )
            hidden = torch.cat((hidden, trace), dim=1)
            valid = torch.cat((valid, trace_ids != PAD), dim=1)
        board_width = board.hidden.shape[1]
        output = self.backbone(hidden, board_width, valid=valid)
        return board, hidden, valid, board_width, output

    def forward(
        self,
        ids=None,
        positions=None,
        trace_ids=None,
        *,
        return_backbone=False,
        trace_positions=None,
    ):
        if ids is None:
            raise ValueError("Sequence policies consume encoded board tokens")
        _, _, _, _, output = self.sequence(ids, positions, trace_ids, trace_positions)
        logits = self.head(output.hidden)
        return (logits, output) if return_backbone else logits

    def trace_logits(self, ids, positions=None, trace_ids=None):
        """Logits at the supervision position of every trace target token."""
        if trace_ids is None:
            raise ValueError("Trace supervision needs trace tokens")
        _, _, valid, board_width, output = self.sequence(ids, positions, trace_ids)
        target = sequence_target_positions(
            valid[:, :board_width].sum(1), trace_ids.shape[1], board_width
        )
        selected = output.hidden.gather(
            1, target.unsqueeze(-1).expand(-1, -1, output.hidden.shape[-1])
        )
        return self.head(selected)

    @torch.no_grad()
    def prefill(self, ids, positions=None):
        """Cache the board prefix; returns the logits predicting the first token."""
        if self.training:
            raise ValueError("Generation requires evaluation mode")
        board = self.board_embedding(ids, positions)
        output = self.backbone.prefill(board.hidden, valid=board.valid)
        last = board.valid.sum(1) - 1
        hidden = output.hidden.gather(
            1, last[:, None, None].expand(-1, 1, output.hidden.shape[-1])
        )
        return self.head(hidden.squeeze(1)), output.cache

    @torch.no_grad()
    def decode(self, tokens, cache):
        """Append trace tokens to a cached prefix; returns next-token logits."""
        if self.training:
            raise ValueError("Generation requires evaluation mode")
        positions = torch.arange(tokens.shape[1], device=tokens.device) + (
            cache.length - cache.prefix_length
        )
        hidden = self._trace_embeddings(
            tokens, positions, self.trace_embedding.weight.dtype
        )
        output = self.backbone.decode(hidden, cache, valid=tokens != PAD)
        return self.head(output.hidden[:, -1]), output.cache
