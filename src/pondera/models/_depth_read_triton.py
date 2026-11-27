"""Gather and differentiate token-local depth attention without stacking sources."""

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(
    v0,
    v1,
    v2,
    v3,
    v4,
    v5,
    r0,
    r1,
    r2,
    r3,
    r4,
    r5,
    weight,
    output,
    lse,
    rows,
    WIDTH: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    SOURCE_COUNT: tl.constexpr,
):
    values = (v0, v1, v2, v3, v4, v5)
    inverse_rms = (r0, r1, r2, r3, r4, r5)
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col = tl.arange(0, BLOCK_WIDTH)
    offset = row[:, None] * WIDTH + col[None, :]
    valid = (row[:, None] < rows) & (col[None, :] < WIDTH)
    query = tl.load(weight + col, col < WIDTH, 0)
    maximum = tl.full((BLOCK_ROWS,), float("-inf"), tl.float32)
    denominator = tl.full((BLOCK_ROWS,), 0, tl.float32)
    numerator = tl.full((BLOCK_ROWS, BLOCK_WIDTH), 0, tl.float32)

    for source in tl.static_range(SOURCE_COUNT):
        value = tl.load(values[source] + offset, valid, 0)
        inverse = tl.load(inverse_rms[source] + row, row < rows, 0)
        score = tl.sum(value * query[None, :], 1) * inverse
        updated_maximum = tl.maximum(maximum, score)
        previous = tl.exp(maximum - updated_maximum)
        current = tl.exp(score - updated_maximum)
        denominator = denominator * previous + current
        numerator = numerator * previous[:, None] + value * current[:, None]
        maximum = updated_maximum

    tl.store(output + offset, numerator / denominator[:, None], valid)
    tl.store(lse + row, maximum + tl.log(denominator), row < rows)


@triton.jit
def _backward(
    v0,
    v1,
    v2,
    v3,
    v4,
    v5,
    r0,
    r1,
    r2,
    r3,
    r4,
    r5,
    weight,
    output,
    lse,
    grad_output,
    g0,
    g1,
    g2,
    g3,
    g4,
    g5,
    grad_weight_tiles,
    rows,
    WIDTH: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
    SOURCE_COUNT: tl.constexpr,
):
    values = (v0, v1, v2, v3, v4, v5)
    inverse_rms = (r0, r1, r2, r3, r4, r5)
    grad_values = (g0, g1, g2, g3, g4, g5)
    program = tl.program_id(0)
    col = tl.arange(0, BLOCK_WIDTH)
    query = tl.load(weight + col, col < WIDTH, 0)
    query_grad = tl.full((BLOCK_ROWS, BLOCK_WIDTH), 0, tl.float32)
    for tile in range(TILES_PER_PROGRAM):
        row = (program * TILES_PER_PROGRAM + tile) * BLOCK_ROWS + tl.arange(
            0, BLOCK_ROWS
        )
        offset = row[:, None] * WIDTH + col[None, :]
        valid = (row[:, None] < rows) & (col[None, :] < WIDTH)
        out = tl.load(output + offset, valid, 0)
        grad = tl.load(grad_output + offset, valid, 0)
        log_normalizer = tl.load(lse + row, row < rows, 0)
        grad_center = tl.sum(grad * out, 1)
        for source in tl.static_range(SOURCE_COUNT):
            value = tl.load(values[source] + offset, valid, 0)
            inverse = tl.load(inverse_rms[source] + row, row < rows, 0)
            dot = tl.sum(value * query[None, :], 1)
            probability = tl.exp(dot * inverse - log_normalizer)
            score_grad = probability * (tl.sum(grad * value, 1) - grad_center)
            dot_grad = score_grad * inverse
            norm_grad = -(dot_grad * dot * inverse * inverse) / WIDTH
            value_grad = (
                probability[:, None] * grad
                + dot_grad[:, None] * query[None, :]
                + norm_grad[:, None] * value
            )
            tl.store(grad_values[source] + offset, value_grad, valid)
            query_grad += dot_grad[:, None] * value

    tl.store(
        grad_weight_tiles + program * WIDTH + col, tl.sum(query_grad, 0), col < WIDTH
    )


def _pointers(tensors):
    # The Ascend launcher accepts individual tensor arguments.
    return (*tensors, *((tensors[0],) * (6 - len(tensors))))


class _DepthRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, *inputs):
        source_count = len(inputs) // 2
        values, inverse_rms = inputs[:source_count], inputs[source_count:]
        output = torch.empty_like(values[0])
        rows, width = output.shape
        lse = output.new_empty(rows)
        block_rows = 4
        _forward[(triton.cdiv(rows, block_rows),)](
            *_pointers(values),
            *_pointers(inverse_rms),
            weight,
            output,
            lse,
            rows,
            width,
            triton.next_power_of_2(width),
            block_rows,
            source_count,
        )
        ctx.save_for_backward(weight, output, lse, *inputs)
        ctx.source_count = source_count
        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight, output, lse, *inputs = ctx.saved_tensors
        source_count = ctx.source_count
        values, inverse_rms = tuple(inputs[:source_count]), tuple(inputs[source_count:])
        rows, width = output.shape
        block_rows, tiles_per_program = 4, 8
        programs = triton.cdiv(rows, block_rows * tiles_per_program)
        grad_values = tuple(torch.empty_like(value) for value in values)
        grad_weight_tiles = output.new_empty((programs, width))
        _backward[(programs,)](
            *_pointers(values),
            *_pointers(inverse_rms),
            weight,
            output,
            lse,
            grad_output.contiguous(),
            *_pointers(grad_values),
            grad_weight_tiles,
            rows,
            width,
            triton.next_power_of_2(width),
            block_rows,
            tiles_per_program,
            source_count,
        )
        # The fused value gradient already includes the derivative through inverse_rms.
        return (
            grad_weight_tiles.sum(0).view_as(weight),
            *grad_values,
            *((None,) * source_count),
        )


def depth_read(sources, weight):
    """Read contiguous FP32 source values paired with their tokenwise inverse RMS."""
    if not 1 <= len(sources) <= 6:
        raise ValueError("Fused depth read supports one to six sources")
    values, inverse_rms = zip(*sources, strict=True)
    return _DepthRead.apply(weight, *values, *inverse_rms)
