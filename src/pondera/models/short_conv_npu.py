"""Fused Ascend decode step for the KDA-family short convolutions."""

import torch
import triton
import triton.language as tl


@triton.jit
def _convolution(x, old, weight, out, new, channels, width: tl.constexpr):
    valid = channels < width
    x0 = tl.load(old + channels, mask=valid, other=0)
    x1 = tl.load(old + width + channels, mask=valid, other=0)
    x2 = tl.load(old + 2 * width + channels, mask=valid, other=0)
    x3 = tl.load(x + channels, mask=valid, other=0)
    w0 = tl.load(weight + 4 * channels, mask=valid, other=0)
    w1 = tl.load(weight + 4 * channels + 1, mask=valid, other=0)
    w2 = tl.load(weight + 4 * channels + 2, mask=valid, other=0)
    w3 = tl.load(weight + 4 * channels + 3, mask=valid, other=0)
    result = (x0.to(tl.float32) * w0 + x1.to(tl.float32) * w1) + (
        x2.to(tl.float32) * w2 + x3.to(tl.float32) * w3
    )
    tl.store(out + channels, result / (1.0 + tl.exp(-result)), mask=valid)
    tl.store(new + channels, x1, mask=valid)
    tl.store(new + width + channels, x2, mask=valid)
    tl.store(new + 2 * width + channels, x3, mask=valid)


@triton.jit
def _qkv_convolution(
    xq,
    xk,
    xv,
    hq,
    hk,
    hv,
    wq,
    wk,
    wv,
    result,
    history,
    width: tl.constexpr,
    block: tl.constexpr,
):
    channels = tl.program_id(0) * block + tl.arange(0, block)
    _convolution(xq, hq, wq, result, history, channels, width)
    _convolution(xk, hk, wk, result + width, history + 3 * width, channels, width)
    _convolution(xv, hv, wv, result + 2 * width, history + 6 * width, channels, width)


def short_convolution_step(projections, weights, histories):
    """Return FP32 convolution outputs and fresh histories for one decode token."""
    x = projections[0]
    width = x.shape[-1]
    if x.dtype not in (torch.float32, torch.bfloat16):
        return None
    if any(
        t.device != x.device or not t.is_contiguous()
        for t in (*projections, *weights, *histories)
    ):
        return None
    if any(p.shape != (1, width) or p.dtype != x.dtype for p in projections):
        return None
    if any(h.shape != (3, width) or h.dtype != x.dtype for h in histories):
        return None
    if any(w.shape != (width, 1, 4) or w.dtype != torch.float32 for w in weights):
        return None

    result = x.new_empty((3, 1, width), dtype=torch.float32)
    history = x.new_empty((3, 3, width))
    _qkv_convolution[(triton.cdiv(width, 64),)](
        *projections,
        *histories,
        *weights,
        result,
        history,
        width=width,
        block=64,
        enable_fp_fusion=False,
    )
    return result.unbind(0), history.unbind(0)
