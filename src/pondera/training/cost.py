"""Analytical dense-matmul FLOPs for board-policy comparisons.

A multiply-add counts as two FLOPs. Padding, embedding lookups, normalization,
activation functions, scalar bias arithmetic, and optimizer updates are excluded.
Logical token lengths, not padded batch widths, define the invariant budget.
Training counts forward plus two backward matmuls for trainable linear/attention
operations.
"""

import numpy as np

from pondera.chess.vocabulary import CHESS_VOCAB_SIZE


def depth_flops(config, lengths, head_exits, *, training=True, repetitions=None):
    """FLOPs for recurrent-depth policies under the module conventions.

    Trunk layers execute once per core repetition; the shared vocabulary head
    applies at every computed exit on the two readout positions. repetitions
    overrides the configured per-core maximum for analytical comparisons at a
    lower executed depth.
    """
    if config.get("architecture") != "depth":
        raise ValueError("This FLOP convention is defined for the depth policy")
    if head_exits < 1:
        raise ValueError("At least one exit must apply the head")
    if repetitions is not None and repetitions < 1:
        raise ValueError("Executed repetitions must be positive")
    lengths = np.asarray(lengths, dtype=np.float64) + 2  # Action and value readouts.
    width, intermediate = config["hidden_size"], config["intermediate_size"]
    layers = sum(
        core["layers"]
        * (
            repetitions
            if repetitions is not None
            else core.get("repetition", {}).get("max_repetitions", 1)
        )
        for core in config["cores"]
    )
    trunk = layers * (
        8 * lengths * width**2
        + 6 * lengths * width * intermediate
        + 4 * lengths**2 * width
    )
    head = head_exits * 2 * (2 * width * CHESS_VOCAB_SIZE)
    return (3 if training else 1) * (trunk + head)


def sequence_flops(config, board_lengths, trace_lengths, *, training=True):
    """FLOPs for the causal sequence policy over actual board and trace lengths.

    The board prefix attends bidirectionally and every trace token attends the
    prefix plus the trace up to itself. A row with ``B`` board atoms and ``T``
    trace tokens therefore has ``B**2 + B*T + T*(T+1)/2`` query-key pairs. The
    vocabulary head is applied at the ``T`` supervised trace positions. Callers
    pass exact non-padding counts: padded positions cost nothing, while tool
    responses and the final move are ordinary attended tokens and do count.
    """
    if config.get("architecture") != "sequence":
        raise ValueError("This FLOP convention is defined for the sequence policy")
    board = np.asarray(board_lengths, dtype=np.float64)
    trace = np.asarray(trace_lengths, dtype=np.float64)
    if board.shape != trace.shape:
        raise ValueError("Board and trace lengths must describe the same rows")
    if np.any(board < 1) or np.any(trace < 1):
        raise ValueError("Every row needs at least one board and one trace token")
    width, intermediate = config["hidden_size"], config["intermediate_size"]
    layers = config["num_blocks"]
    tokens = board + trace
    pairs = board**2 + board * trace + trace * (trace + 1) / 2
    trunk = layers * (
        2 * tokens * (4 * width**2 + 3 * width * intermediate) + 4 * width * pairs
    )
    head = 2 * trace * width * CHESS_VOCAB_SIZE
    return (3 if training else 1) * (trunk + head)
