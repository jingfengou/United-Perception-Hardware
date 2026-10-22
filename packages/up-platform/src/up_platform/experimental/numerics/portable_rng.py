"""GPU-only Philox stream; imported only by the explicit portable profile."""
import triton
import triton.language as tl


@triton.jit
def _uniform(output, count, seed, start, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offset = index.to(tl.uint64) + start.to(tl.uint64)
    value = tl.rand(seed, offset, n_rounds=10)
    tl.store(output + index, value, index < count)


def uniform(shape, device, seed, offset):
    import torch
    output = torch.empty(shape, dtype=torch.float32, device=device)
    if output.numel():
        _uniform[(triton.cdiv(output.numel(), 256),)](
            output, output.numel(), seed, offset, BLOCK=256)
    return output
