"""A causal transformer with weight sharing across depth in the middle blocks."""

import random
from itertools import chain, repeat

import torch
from torch.utils.checkpoint import checkpoint

from pondera.models.delta import GatedDeltaAttention, KimiDeltaAttention
from pondera.models.vanilla import Block, CausalSelfAttention, RMSNorm, VanillaModel


class LoopedModel(VanillaModel):
    architecture = "looped"

    def __init__(
        self, *, loop_repetitions=3, mixer="softmax", softmax_every=0, **kwargs
    ):
        if kwargs["num_blocks"] < 5:
            raise ValueError("A looped model needs two blocks on each side of its core")
        if loop_repetitions < 1:
            raise ValueError("Require at least one repetition")
        mixers = {
            "softmax": CausalSelfAttention,
            "gdn": GatedDeltaAttention,
            "kda": KimiDeltaAttention,
        }
        if mixer not in mixers:
            raise ValueError(f"Unknown loop mixer: {mixer}")
        if not 0 <= softmax_every <= kwargs["num_blocks"]:
            raise ValueError("softmax_every must be zero or a valid layer interval")
        if mixer == "softmax" and softmax_every:
            raise ValueError("softmax_every requires a linear mixer")
        self.attention_cls = mixers[mixer]
        self.softmax_every = softmax_every
        super().__init__(**kwargs)
        self.loop_repetitions = loop_repetitions
        self.training_repetitions = loop_repetitions

    def attention_type(self, index):
        if self.softmax_every and (index + 1) % self.softmax_every == 0:
            return CausalSelfAttention
        return self.attention_cls

    def select_training_depth(self, seed, update):
        self.training_repetitions = random.Random(f"{seed}:loop:{update}").randint(
            1, self.loop_repetitions
        )
        return self.training_repetitions

    def execution_blocks(self):
        repetitions = (
            self.training_repetitions if self.training else self.loop_repetitions
        )
        core = self.blocks[2:-2]
        return chain(
            self.blocks[:2],
            chain.from_iterable(repeat(core, repetitions)),
            self.blocks[-2:],
        )


class SandwichBlock(Block):
    """Normalize both residual outputs, following recurrent-depth Section 3.2."""

    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__(hidden_size, *args, **kwargs)
        self.norm_after_attention = RMSNorm(hidden_size)
        self.norm_after_ffn = RMSNorm(hidden_size)

    def forward(self, x, prepared):
        x = self.norm_after_attention(
            x + self.attention(self.norm_attention(x), prepared)
        )
        return self.norm_after_ffn(x + self._ffn(self.norm_ffn(x)))

    def forward_cached(self, x, cu_seqlens, cache=None):
        attended, new_cache = self.attention.forward_cached(
            self.norm_attention(x), cu_seqlens, cache
        )
        x = self.norm_after_attention(x + attended)
        return self.norm_after_ffn(x + self._ffn(self.norm_ffn(x))), new_cache


class RecurrentDepthModel(LoopedModel):
    """Inject the prelude at every normalized core iteration.

    Uses additive input injection and a zero initial state. This deterministic
    small-model variant preserves packed and incremental causal execution;
    it does not reproduce the paper's random initialization or depth sampler.
    """

    architecture = "recurrent"
    block_cls = SandwichBlock

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm_core = RMSNorm(kwargs["hidden_size"])

    def _iterate(self, x, apply):
        for block in self.blocks[:2]:
            x = apply(block, x)
        embedded = x
        state = torch.zeros_like(embedded)
        repetitions = (
            self.training_repetitions if self.training else self.loop_repetitions
        )
        for _ in range(repetitions):
            state = state + embedded
            for block in self.blocks[2:-2]:
                state = apply(block, state)
            state = self.norm_core(state)
        for block in self.blocks[-2:]:
            state = apply(block, state)
        return self.head(self.norm(state))

    def forward(self, input_ids, cu_seqlens):
        prepared = self.prepare(cu_seqlens, input_ids.device)

        def apply(block, x):
            metadata = prepared[type(block.attention)]
            if self.training and self.activation_checkpointing:
                return checkpoint(block, x, metadata, use_reentrant=False)
            return block(x, metadata)

        return self._iterate(self.token_embedding(input_ids), apply)

    def forward_cached(self, input_ids, cu_seqlens, caches=None):
        new_caches = []

        def apply(block, x):
            cache = None if caches is None else caches[len(new_caches)]
            x, new_cache = block.forward_cached(x, cu_seqlens, cache)
            new_caches.append(new_cache)
            return x

        logits = self._iterate(self.token_embedding(input_ids), apply)
        return logits, new_caches
