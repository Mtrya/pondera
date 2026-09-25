"""Analytical dense-matmul FLOPs for fixed-depth board-policy comparisons.

A multiply-add counts as two FLOPs. Padding, embedding lookups, normalization,
activation functions, scalar bias arithmetic, and optimizer updates are excluded.
Logical token lengths, not padded batch widths, define the invariant budget.
Training counts forward plus two backward matmuls for trainable linear/attention
operations; the unsupervised value projection incurs its forward cost only.
"""

import numpy as np


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
