"""A causal transformer with weight sharing across depth in the middle blocks."""

import random

import torch
from torch.utils.checkpoint import checkpoint

from pondera.models.delta import GatedDeltaAttention, KimiDeltaAttention
from pondera.models.vanilla import Block, CausalSelfAttention, RMSNorm, VanillaModel


class LoopedModel(VanillaModel):
    architecture = "looped"

    def __init__(
        self,
        *,
        loop_repetitions=3,
        mixer="softmax",
        softmax_every=0,
        scale_residuals=False,
        **kwargs,
    ):
        if kwargs["num_blocks"] < 5:
            raise ValueError("A looped model needs two blocks on each side of its core")
        if loop_repetitions < 1:
            raise ValueError("Require at least one repetition")
        if type(scale_residuals) is not bool or (
            scale_residuals and self.architecture != "looped"
        ):
            raise ValueError("Residual scaling requires the plain looped architecture")
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
        self.scale_residuals = scale_residuals

    def attention_type(self, index):
        if self.softmax_every and (index + 1) % self.softmax_every == 0:
            return CausalSelfAttention
        return self.attention_cls

    def select_training_depth(self, seed, update):
        self.training_repetitions = random.Random(f"{seed}:loop:{update}").randint(
            1, self.loop_repetitions
        )
        return self.training_repetitions

    def _iterate(self, x, apply):
        for block in self.blocks[:2]:
            x = apply(block, x)
        repetitions = (
            self.training_repetitions if self.training else self.loop_repetitions
        )
        scale = 1.0 / repetitions if self.scale_residuals else 1.0
        for _ in range(repetitions):
            for block in self.blocks[2:-2]:
                x = apply(block, x, scale)
        for block in self.blocks[-2:]:
            x = apply(block, x)
        return self.head(self.norm(x))

    def forward(self, input_ids, cu_seqlens):
        prepared = self.prepare(cu_seqlens, input_ids.device)

        def apply(block, x, scale=1.0):
            metadata = prepared[type(block.attention)]
            options = {} if scale == 1.0 else {"residual_scale": scale}
            if self.training and self.activation_checkpointing:
                return checkpoint(block, x, metadata, use_reentrant=False, **options)
            return block(x, metadata, **options)

        return self._iterate(self.token_embedding(input_ids), apply)

    def forward_cached(self, input_ids, cu_seqlens, caches=None):
        new_caches = []

        def apply(block, x, scale=1.0):
            cache = None if caches is None else caches[len(new_caches)]
            options = {} if scale == 1.0 else {"residual_scale": scale}
            x, new_cache = block.forward_cached(x, cu_seqlens, cache, **options)
            new_caches.append(new_cache)
            return x

        logits = self._iterate(self.token_embedding(input_ids), apply)
        return logits, new_caches


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
