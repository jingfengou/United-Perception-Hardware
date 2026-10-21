"""Explicit GPU FP64 arithmetic for the opt-in numerical profile."""
import math

import torch
import triton
import triton.language as tl

from ..errors import CapabilityError


@triton.jit
def _matrix(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            AB: tl.constexpr, AM: tl.constexpr, AK: tl.constexpr,
            BB: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
            TM: tl.constexpr, TN: tl.constexpr):
    batch = tl.program_id(2)
    m = tl.program_id(0) * TM + tl.arange(0, TM)
    n = tl.program_id(1) * TN + tl.arange(0, TN)
    value = tl.full((TM, TN), 0, tl.float64)
    # Explicit FMA and sequential K order avoid backend-dependent reduction trees.
    for k in range(K):
        a = tl.load(A + batch * AB + m[:, None] * AM + k * AK, m[:, None] < M, 0)
        b = tl.load(B + batch * BB + k * BK + n[None, :] * BN, n[None, :] < N, 0)
        value = tl.fma(a, b, value)
    tl.store(C + batch * M * N + m[:, None] * N + n[None, :], value,
             (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _row_sum(X, Y, COLS: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    column = part * BLOCK + lane
    value = tl.load(X + row * COLS + column, column < COLS, 0)
    # The logical butterfly is explicit, rather than left to tl.sum lowering.
    for bit in tl.static_range(0, 8):
        peer = tl.gather(value, lane ^ (1 << bit), axis=0)
        value = value + peer
    tl.store(Y + row * PARTS + part + lane * 0, value, lane == 0)


def _check(value):
    if value.device.type != "cuda" or value.dtype != torch.float64:
        raise CapabilityError("portable_v1 shared arithmetic requires GPU FP64 inputs")


def matrix(a, b):
    _check(a)
    _check(b)
    if a.device != b.device or a.ndim not in (2, 3) or b.ndim != a.ndim:
        raise CapabilityError("portable_v1 matrix requires aligned 2D or 3D GPU operands")
    if a.shape[-1] != b.shape[-2] or a.ndim == 3 and a.shape[0] != b.shape[0]:
        raise ValueError("Incompatible matrix shapes")
    batch = a.shape[0] if a.ndim == 3 else 1
    m, k = a.shape[-2:]
    n = b.shape[-1]
    output = torch.empty((*a.shape[:-2], m, n), device=a.device, dtype=a.dtype)
    if batch and m and n:
        _matrix[(triton.cdiv(m, 4), triton.cdiv(n, 32), batch)](
            a, b, output, m, n, k,
            a.stride(0) if a.ndim == 3 else 0, *a.stride()[-2:],
            b.stride(0) if b.ndim == 3 else 0, *b.stride()[-2:],
            TM=4, TN=32, num_warps=4, enable_fp_fusion=False)
    return output


def sum_rows(value):
    _check(value)
    if value.ndim != 2:
        raise ValueError("Row reduction requires a matrix")
    value = value.contiguous()
    rows, columns = value.shape
    if not columns:
        return value.new_zeros((rows,))
    while columns > 1:
        parts = triton.cdiv(columns, 256)
        result = torch.empty((rows, parts), dtype=value.dtype, device=value.device)
        if rows:
            _row_sum[(rows, parts)](value, result, columns, parts, BLOCK=256,
                                    num_warps=4, enable_fp_fusion=False)
        value, columns = result, parts
    return value[:, 0]


def reduce_sum(value, dims=None, keepdim=False):
    _check(value)
    if isinstance(dims, int):
        dims = (dims,)
    if value.ndim == 0:
        if dims not in (None, (), [], (0,), (-1,)):
            raise ValueError("Invalid scalar reduction dimension")
        return value
    dims = tuple(range(value.ndim)) if dims is None or len(dims) == 0 else tuple(d % value.ndim for d in dims)
    if len(set(dims)) != len(dims):
        raise ValueError("Repeated reduction dimension")
    kept = tuple(d for d in range(value.ndim) if d not in dims)
    shape = tuple(value.shape[d] for d in kept)
    columns = math.prod(value.shape[d] for d in dims)
    matrix_value = value.permute(*kept, *dims).contiguous().reshape(math.prod(shape), columns)
    result = sum_rows(matrix_value).reshape(shape)
    if keepdim:
        result = result.reshape(tuple(1 if d in dims else value.shape[d] for d in range(value.ndim)))
    return result


def channel_sum(value):
    return reduce_sum(value, (0, *range(2, value.ndim)))
