"""Shared L1 pair distances and their complete first-order backward."""
import torch


def reduce_sum(value, dims=None, keepdim=False):
    from .portable_math import reduce_sum as implementation
    return implementation(value, dims, keepdim)


def l1_distance(x1, x2, p=1., compute_mode=None):
    if p != 1:
        raise ValueError("This implementation requires the L1 distance")
    difference = x1.unsqueeze(-2) - x2.unsqueeze(-3)
    return reduce_sum(difference.abs(), (-1,))


def l1_backward(grad, x1, x2, p, cdist):
    if p != 1:
        raise ValueError("This implementation requires the L1 distance")
    difference = x1.unsqueeze(-2) - x2.unsqueeze(-3)
    result = reduce_sum(grad.unsqueeze(-1) * difference.sign(), (-2,))
    if result.shape != x1.shape:
        padding = result.ndim - x1.ndim
        target = (1,) * padding + tuple(x1.shape)
        dims = tuple(i for i, (a, b) in enumerate(zip(result.shape, target)) if a != b)
        result = (reduce_sum(result, dims, True) if dims else result).reshape_as(x1)
    return result


def vector_norm(value, ord=2, dim=None, keepdim=False, dtype=None):
    if ord != 2:
        raise ValueError("This implementation requires the L2 norm")
    output = torch.sqrt(reduce_sum(value.square(), dim, keepdim))
    return output if dtype is None else output.to(dtype)
