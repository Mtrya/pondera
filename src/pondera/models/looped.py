"""A causal transformer with weight sharing across depth in the middle blocks."""

import math
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from pondera.models.delta import GatedDeltaAttention, KimiDeltaAttention
from pondera.models.vanilla import Block, CausalSelfAttention, RMSNorm, VanillaModel


class InputInjection(nn.Module):
    """Learned diagonal carry and input projection, optionally using OrthoInj."""

    def __init__(self, hidden_size, *, orthogonal):
        super().__init__()
        self.orthogonal = orthogonal
        self.input_adapter = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.input_adapter.weight)
        step = 0.5 * math.log(5)
        self.dt_bias = nn.Parameter(
            torch.full((hidden_size,), step + math.log(-math.expm1(-step)))
        )
        self.A_log = nn.Parameter(torch.zeros(hidden_size))

    def prepare(self, embedded):
        step = F.softplus(self.dt_bias).to(embedded.dtype)
        decay = torch.exp(-step * self.A_log.exp().to(embedded.dtype))
        injected = self.input_adapter(embedded).to(embedded.dtype) * step
        return injected, decay

    def forward(self, state, prepared):
        injected, decay = prepared
        retained = state * decay
        if self.orthogonal:
            dtype = torch.float64 if retained.dtype == torch.float64 else torch.float32
            value, anchor = retained.to(dtype), injected.to(dtype)
            coefficient = (value * anchor).sum(dim=-1, keepdim=True) / (
                anchor.square().sum(dim=-1, keepdim=True) + 1e-6
            )
            retained = (value - coefficient * anchor).to(retained.dtype)
        return retained + injected


class LoopedModel(VanillaModel):
    architecture = "looped"

    def __init__(
        self,
        *,
        loop_repetitions=3,
        mixer="softmax",
        softmax_every=0,
        scale_residuals=False,
        input_injection="none",
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
        if input_injection not in {"none", "decay", "orthogonal"} or (
            input_injection != "none" and self.architecture != "looped"
        ):
            raise ValueError(
                "Input injection requires a plain looped model and a known mode"
            )
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
        self.injection = (
            None
            if input_injection == "none"
            else InputInjection(
                kwargs["hidden_size"], orthogonal=input_injection == "orthogonal"
            )
        )

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
        if self.injection is not None:
            injected = self.injection.prepare(x)
            x = torch.zeros_like(x)
        for _ in range(repetitions):
            if self.injection is not None:
                x = self.injection(x, injected)
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
