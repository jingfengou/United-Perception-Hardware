"""Fixed FP32 GEMM; no autotuning, split-K, atomics or reduced-precision dot."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          AM: tl.constexpr, AK: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
          TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr):
    m = tl.program_id(0) * TM + tl.arange(0, TM)
    n = tl.program_id(1) * TN + tl.arange(0, TN)
    offsets = tl.arange(0, TK)
    value = tl.full((TM, TN), 0, tl.float32)
    for start in range(tl.cdiv(K, TK)):
        k = start * TK + offsets
        a = tl.load(A + m[:, None] * AM + k[None, :] * AK, (m[:, None] < M) & (k[None, :] < K), 0)
        b = tl.load(B + k[:, None] * BK + n[None, :] * BN, (k[:, None] < K) & (n[None, :] < N), 0)
        value = tl.dot(a, b, value, input_precision="ieee")
    tl.store(C + m[:, None] * N + n[None, :], value, (m[:, None] < M) & (n[None, :] < N))


def launch(a, b):
    m, k = a.shape
    n = b.shape[1]
    output = torch.empty((m, n), device=a.device, dtype=torch.float32)
    if m and n:
        with torch.cuda.device(a.device):
            _gemm[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
                a, b, output, m, n, k, *a.stride(), *b.stride(),
                TM=32, TN=32, TK=32, num_warps=4, enable_fp_fusion=False)
    return output
