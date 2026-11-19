"""Normalized Möbius scans and reverse affine scans for packed covariance."""

import torch
import triton
import triton.language as tl


@triton.jit
def _compose(a, b, c, d, e, f, g, h):
    # The later map multiplies the earlier map on the left.
    aa, bb = e * a + f * c, e * b + f * d
    cc, dd = g * a + h * c, g * b + h * d
    scale = tl.maximum(
        tl.maximum(tl.abs(aa), tl.abs(bb)), tl.maximum(tl.abs(cc), tl.abs(dd))
    )
    scale = tl.maximum(scale, 1e-30)
    return aa / scale, bb / scale, cc / scale, dd / scale


@triton.jit
def _affine(a, b, c, d):
    return c * a, c * b + d


@triton.jit(do_not_specialize=["row_offset", "column_offset"])
def _forward_tiles(
    A,
    W,
    U,
    Starts,
    Ends,
    Documents,
    Offsets,
    Initial,
    Maps,
    Carry,
    Output,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    EMIT: tl.constexpr,
    row_offset,
    column_offset,
):
    chunk = tl.program_id(0) + row_offset
    lane = (tl.program_id(1) + column_offset) * BS + tl.arange(0, BS)
    start, end = tl.load(Starts + chunk), tl.load(Ends + chunk)
    row = tl.arange(0, BT)
    token = start + row
    address = token[:, None] * S + lane[None, :]
    mask = (token[:, None] < end) & (lane[None, :] < S)
    a = tl.load(A + address, mask, other=1).to(tl.float32)
    w = tl.load(W + address, mask, other=0).to(tl.float32)
    u = tl.load(U + address, mask, other=0).to(tl.float32)
    aa, bb, cc, dd = tl.associative_scan((a, w, u * a, 1 + u * w), 0, _compose)
    if EMIT:
        doc = tl.load(Documents + chunk)
        first = tl.load(Offsets + doc)
        prior = tl.load(Initial + doc * S + lane, lane < S, other=0)
        previous = tl.load(
            Carry + (chunk - 1) * S + lane, (chunk > first) & (lane < S), other=0
        )
        incoming = tl.where(chunk == first, prior, previous)
        state = (aa * incoming[None, :] + bb) / (cc * incoming[None, :] + dd)
        tl.store(Output + address, state, mask)
    else:
        last = row[:, None] == BT - 1
        base = chunk * 4 * S + lane
        tl.store(Maps + base, tl.sum(tl.where(last, aa, 0), 0), lane < S)
        tl.store(Maps + base + S, tl.sum(tl.where(last, bb, 0), 0), lane < S)
        tl.store(Maps + base + 2 * S, tl.sum(tl.where(last, cc, 0), 0), lane < S)
        tl.store(Maps + base + 3 * S, tl.sum(tl.where(last, dd, 0), 0), lane < S)


@triton.jit(do_not_specialize=["row_offset", "column_offset"])
def _forward_carry(
    Maps,
    Offsets,
    Initial,
    Carry,
    S: tl.constexpr,
    NT: tl.constexpr,
    row_offset,
    column_offset,
):
    doc, lane = tl.program_id(0) + row_offset, tl.program_id(1) + column_offset
    start, end = tl.load(Offsets + doc), tl.load(Offsets + doc + 1)
    prior = tl.load(Initial + doc * S + lane)
    row = tl.arange(0, NT)
    for offset in range(start, end, NT):
        chunk = offset + row
        base = chunk * 4 * S + lane
        a = tl.load(Maps + base, chunk < end, other=1)
        b = tl.load(Maps + base + S, chunk < end, other=0)
        c = tl.load(Maps + base + 2 * S, chunk < end, other=0)
        d = tl.load(Maps + base + 3 * S, chunk < end, other=1)
        a, b, c, d = tl.associative_scan((a, b, c, d), 0, _compose)
        state = (a * prior + b) / (c * prior + d)
        tl.store(Carry + chunk * S + lane, state, chunk < end)
        prior = tl.sum(tl.where(row == NT - 1, state, 0), 0)


@triton.jit(do_not_specialize=["row_offset", "column_offset"])
def _backward_tiles(
    A,
    W,
    U,
    Initial,
    Output,
    DOutput,
    Starts,
    Ends,
    Documents,
    Offsets,
    Maps,
    Carry,
    DA,
    DW,
    DU,
    DInitial,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    EMIT: tl.constexpr,
    row_offset,
    column_offset,
):
    chunk = tl.program_id(0) + row_offset
    lane = (tl.program_id(1) + column_offset) * BS + tl.arange(0, BS)
    start, end = tl.load(Starts + chunk), tl.load(Ends + chunk)
    doc = tl.load(Documents + chunk)
    first_chunk, last_chunk = tl.load(Offsets + doc), tl.load(Offsets + doc + 1)
    doc_start = tl.load(Starts + first_chunk)
    row = tl.arange(0, BT)
    token = start + row
    address = token[:, None] * S + lane[None, :]
    mask = (token[:, None] < end) & (lane[None, :] < S)
    a = tl.load(A + address, mask, other=1).to(tl.float32)
    w = tl.load(W + address, mask, other=0).to(tl.float32)
    u = tl.load(U + address, mask, other=0).to(tl.float32)
    output = tl.load(Output + address, mask, other=0)
    previous_address = tl.maximum(token - 1, doc_start)[:, None] * S + lane[None, :]
    previous = tl.load(
        Output + previous_address, mask & (token[:, None] > doc_start), other=0
    )
    prior = tl.load(Initial + doc * S + lane, lane < S, other=0)
    previous = tl.where(token[:, None] == doc_start, prior[None, :], previous)
    z = a * previous + w
    inverse = 1 / (1 + u * z)
    derivative = a * inverse * inverse
    grad = tl.load(DOutput + address, mask, other=0).to(tl.float32)
    aa, bb = tl.associative_scan(
        (derivative, derivative * grad), 0, _affine, reverse=True
    )
    if EMIT:
        incoming = tl.load(
            Carry + (chunk + 1) * S + lane,
            (chunk + 1 < last_chunk) & (lane < S),
            other=0,
        )
        before = aa * incoming[None, :] + bb
        next_row = tl.minimum(row + 1, BT - 1)
        after = tl.gather(before, tl.broadcast_to(next_row[:, None], (BT, BS)), 0)
        after = tl.where(row[:, None] == BT - 1, incoming[None, :], after)
        adjoint = grad + after
        dz = adjoint * inverse * inverse
        tl.store(DA + address, dz * previous, mask)
        tl.store(DW + address, dz, mask)
        tl.store(DU + address, -adjoint * output * output, mask)
        if chunk == first_chunk:
            value = tl.sum(tl.where(row[:, None] == 0, before, 0), 0)
            tl.store(DInitial + doc * S + lane, value, lane < S)
    else:
        first = row[:, None] == 0
        base = chunk * 2 * S + lane
        tl.store(Maps + base, tl.sum(tl.where(first, aa, 0), 0), lane < S)
        tl.store(Maps + base + S, tl.sum(tl.where(first, bb, 0), 0), lane < S)


@triton.jit(do_not_specialize=["row_offset", "column_offset"])
def _backward_carry(
    Maps,
    Offsets,
    Carry,
    S: tl.constexpr,
    NT: tl.constexpr,
    row_offset,
    column_offset,
):
    doc, lane = tl.program_id(0) + row_offset, tl.program_id(1) + column_offset
    start, end = tl.load(Offsets + doc), tl.load(Offsets + doc + 1)
    row = tl.arange(0, NT)
    incoming = 0.0
    tiles = tl.cdiv(end - start, NT)
    for tile in range(tiles):
        chunk = start + (tiles - tile - 1) * NT + row
        a = tl.load(Maps + chunk * 2 * S + lane, chunk < end, other=1)
        b = tl.load(Maps + chunk * 2 * S + S + lane, chunk < end, other=0)
        a, b = tl.associative_scan((a, b), 0, _affine, reverse=True)
        state = a * incoming + b
        tl.store(Carry + chunk * S + lane, state, chunk < end)
        incoming = tl.sum(tl.where(row == 0, state, 0), 0)


def _launch(kernel, grid, args):
    # Ascend requires bounded launches even when both grid dimensions fit.
    budget = 4096 if args[0].device.type == "npu" else grid[0] * grid[1]
    columns = min(grid[1], budget)
    rows = max(1, budget // columns)
    for row in range(0, grid[0], rows):
        for column in range(0, grid[1], columns):
            shape = (min(rows, grid[0] - row), min(columns, grid[1] - column))
            kernel[shape](*args, row, column)


class _Covariance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, noise, information, initial, prepared):
        starts, ends, docs, offsets, maximum = prepared
        chunks, width = starts.numel(), a.shape[1]
        maps = a.new_empty(chunks, 4, width)
        carry = a.new_empty(chunks, width)
        output = torch.empty_like(a)
        grid = (chunks, triton.cdiv(width, 8))
        args = (
            a,
            noise,
            information,
            starts,
            ends,
            docs,
            offsets,
            initial,
            maps,
            carry,
            output,
        )
        _launch(_forward_tiles, grid, (*args, width, 64, 8, False))
        _launch(
            _forward_carry,
            (initial.shape[0], width),
            (
                maps,
                offsets,
                initial,
                carry,
                width,
                min(512, triton.next_power_of_2(maximum)),
            ),
        )
        _launch(_forward_tiles, grid, (*args, width, 64, 8, True))
        ctx.save_for_backward(a, noise, information, initial, output)
        ctx.prepared = prepared
        return output

    @staticmethod
    def backward(ctx, grad):
        a, noise, information, initial, output = ctx.saved_tensors
        starts, ends, docs, offsets, maximum = ctx.prepared
        chunks, width = starts.numel(), a.shape[1]
        maps = a.new_empty(chunks, 2, width)
        carry = a.new_empty(chunks, width)
        da, dw, du = (torch.empty_like(a) for _ in range(3))
        dinitial = torch.empty_like(initial)
        args = (
            a,
            noise,
            information,
            initial,
            output,
            grad.contiguous(),
            starts,
            ends,
            docs,
            offsets,
            maps,
            carry,
            da,
            dw,
            du,
            dinitial,
        )
        grid = (chunks, triton.cdiv(width, 8))
        _launch(_backward_tiles, grid, (*args, width, 64, 8, False))
        _launch(
            _backward_carry,
            (initial.shape[0], width),
            (maps, offsets, carry, width, min(512, triton.next_power_of_2(maximum))),
        )
        _launch(_backward_tiles, grid, (*args, width, 64, 8, True))
        return da, dw, du, dinitial, None


def covariance(a, noise, information, initial, prepared):
    return _Covariance.apply(
        a.float().contiguous(),
        noise.float().contiguous(),
        information.float().contiguous(),
        initial.float().contiguous(),
        prepared,
    )
