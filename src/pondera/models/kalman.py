"""Packed covariance recurrence for Kalman Delta Networks.

The positive update p <- (a*p+w)/(1+u*(a*p+w)) is a Möbius map.
Reference: https://arxiv.org/abs/2609.07816.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from pondera.models.delta import delta_attention
from pondera.models.memory import HybridModel, RecurrentAttention
from pondera.models.vanilla import RMSNorm

CHUNK_SIZE = 64


def prepare_covariance(cu_seqlens, device):
    boundaries = torch.as_tensor(cu_seqlens, device="cpu").tolist()
    starts, ends, documents, offsets = [], [], [], [0]
    for document, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        for chunk in range(start, end, CHUNK_SIZE):
            starts.append(chunk)
            ends.append(min(chunk + CHUNK_SIZE, end))
            documents.append(document)
        offsets.append(len(starts))
    maximum = max(b - a for a, b in zip(offsets[:-1], offsets[1:]))
    arrays = tuple(
        torch.tensor(values, dtype=torch.int64, device=device)
        for values in (starts, ends, documents, offsets)
    )
    return (*arrays, maximum)


def kalman_covariance(a, noise, information, initial, cu_seqlens, prepared=None):
    """Return posterior covariance [T,S], with an independent [N,S] prior."""
    if a.device.type in {"cuda", "npu"}:
        from pondera.models._kalman_triton import covariance

        if prepared is None:
            prepared = prepare_covariance(cu_seqlens, a.device)
        return covariance(a, noise, information, initial, prepared)
    boundaries = torch.as_tensor(cu_seqlens, device="cpu").tolist()
    results = []
    for document, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        state = initial[document]
        for position in range(start, end):
            prediction = a[position] * state + noise[position]
            state = prediction / (1 + information[position] * prediction)
            results.append(state)
    return torch.stack(results)


def kalman_attention(
    q,
    k,
    v,
    g,
    process_noise,
    observation_noise,
    prepared,
    initial_covariance,
    initial_memory=None,
    output_final=False,
):
    """Normalized Q/K and channel log-decay; cache memory and head covariance."""
    cu_cpu, cu_device = prepared
    posterior = kalman_covariance(
        (2 * g).exp().mean(-1),
        process_noise,
        k.float().square().sum(-1) / observation_noise,
        initial_covariance,
        cu_cpu,
    )
    # With information scale equal to key width, beta is posterior / r.
    beta = posterior / observation_noise
    output, memory = delta_attention(
        q,
        k,
        v,
        g,
        beta,
        prepared,
        initial_memory,
        output_final,
        channel_decay=True,
        normalize=False,
    )
    state = (memory, posterior[cu_device[1:] - 1]) if output_final else None
    return output, state


class KalmanDeltaAttention(RecurrentAttention):
    """Isotropic Kalman uncertainty with channel-decayed associative memory."""

    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__(hidden_size, num_heads)
        self.decay = nn.Sequential(
            nn.Linear(hidden_size, self.head_dim, bias=False),
            nn.Linear(self.head_dim, hidden_size, bias=False),
        )
        self.A_log = nn.Parameter(torch.empty(num_heads).uniform_(1, 16).log())
        dt = torch.exp(
            torch.empty(hidden_size).uniform_(math.log(0.001), math.log(0.1))
        )
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, self.head_dim, bias=False),
            nn.Linear(self.head_dim, hidden_size, bias=True),
        )
        self.norm_output = RMSNorm(self.head_dim, eps=1e-5)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.observation_noise = nn.Linear(hidden_size, num_heads, bias=True)
        self.process_noise = nn.Linear(hidden_size, num_heads, bias=True)
        self.initial_precision = nn.Parameter(
            torch.full((num_heads,), math.log(math.expm1(0.9)))
        )

    def _memory(self, vectors, x, prepared, cache=None, output_final_state=False):
        q, k, v = (vector.view(-1, self.num_heads, self.head_dim) for vector in vectors)
        q, k = (
            (
                vector.float()
                * torch.rsqrt(vector.float().square().sum(-1, keepdim=True) + 1e-6)
            ).to(vector.dtype)
            for vector in (q, k)
        )
        g = F.softplus(self.decay(x).float() + self.dt_bias)
        g = -self.A_log.exp()[None, :, None] * g.view(-1, self.num_heads, self.head_dim)
        noise = F.softplus(self.process_noise(x).float())
        observation = F.softplus(self.observation_noise(x).float()) + 0.01
        cu_cpu, _ = prepared
        memory, covariance = (None, None) if cache is None else cache
        if covariance is None:
            covariance = (F.softplus(self.initial_precision) + 0.1).reciprocal()
            covariance = covariance.expand(len(cu_cpu) - 1, -1).contiguous()
        return kalman_attention(
            q,
            k,
            v,
            g,
            noise,
            observation,
            prepared,
            covariance,
            memory,
            output_final_state,
        )

    def _output(self, x, memory):
        gate = self.gate(x).sigmoid().view_as(memory)
        return self.o((self.norm_output(memory) * gate).reshape_as(x))


class KalmanDeltaModel(HybridModel):
    architecture = "kdn"
    attention_cls = KalmanDeltaAttention
