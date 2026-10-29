"""Analytical dense-matmul FLOPs for fixed-depth board-policy comparisons.

A multiply-add counts as two FLOPs. Padding, embedding lookups, normalization,
activation functions, scalar bias arithmetic, and optimizer updates are excluded.
Logical token lengths, not padded batch widths, define the invariant budget.
Training counts forward plus two backward matmuls for trainable linear/attention
operations; the unsupervised value projection incurs its forward cost only.
"""

import numpy as np

from pondera.chess.tokens import CHESS_VOCAB_SIZE


def policy_flops(config, lengths, *, training=True):
    if config.get("architecture") != "base":
        raise ValueError(
            "This FLOP convention is defined for the fixed-depth base policy"
        )
    lengths = np.asarray(lengths, dtype=np.float64) + 2  # Action and value readouts.
    width, intermediate, layers = (
        config[key] for key in ("hidden_size", "intermediate_size", "num_blocks")
    )
    trunk = layers * (
        8 * lengths * width**2
        + 6 * lengths * width * intermediate
        + 4 * lengths**2 * width
    )
    policy = 2 * width * config.get("possible_moves", 1969)
    value = 2 * width
    return (3 if training else 1) * (trunk + policy) + value


def depth_flops(config, lengths, head_exits, *, training=True, repetitions=None):
    """FLOPs for recurrent-depth policies under the policy_flops conventions.

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
