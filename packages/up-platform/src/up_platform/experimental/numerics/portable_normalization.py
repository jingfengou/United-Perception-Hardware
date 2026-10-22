"""Shared forward/backward normalization formulas; reductions remain on GPU."""
import math

import torch

from up_platform.errors import CapabilityError


def reduce_sum(value, dims=None, keepdim=False):
    from .portable_math import reduce_sum as implementation
    return implementation(value, dims, keepdim)


def channel_sum(value):
    return reduce_sum(value, (0, *range(2, value.ndim)))


def batch_norm(x, weight, bias, running_mean, running_var, training, momentum, eps):
    shape = (1, x.shape[1], *((1,) * (x.ndim - 2)))
    count = x.numel() // x.shape[1]
    if training and count <= 1:
        raise CapabilityError("portable_v1 BatchNorm requires more than one value per channel")
    if training:
        mean = channel_sum(x) / count
        centered = x - mean.reshape(shape)
        variance = channel_sum(centered.square()) / count
        inverse = torch.rsqrt(variance + eps)
        if running_mean is not None:
            running_mean.copy_(running_mean * (1 - momentum) + mean * momentum)
        if running_var is not None:
            running_var.copy_(running_var * (1 - momentum) + variance * (count / (count - 1)) * momentum)
    else:
        mean = running_mean
        inverse = torch.rsqrt(running_var + eps)
        centered = x - mean.reshape(shape)
    output = centered * inverse.reshape(shape)
    if weight is not None:
        output = output * weight.reshape(shape)
    if bias is not None:
        output = output + bias.reshape(shape)
    empty = x.new_empty((0,))
    return output, mean if training else empty, inverse if training else empty


def batch_norm_backward(gradient, x, weight, running_mean, running_var,
                        saved_mean, saved_inverse, training, eps, output_mask):
    shape = (1, x.shape[1], *((1,) * (x.ndim - 2)))
    count = x.numel() // x.shape[1]
    mean = saved_mean if training else running_mean
    inverse = saved_inverse if training else torch.rsqrt(running_var + eps)
    normalized = (x - mean.reshape(shape)) * inverse.reshape(shape)
    total = channel_sum(gradient)
    projected = channel_sum(gradient * normalized)
    dx = gradient
    if training:
        dx = dx - total.reshape(shape) / count - normalized * projected.reshape(shape) / count
    dx = dx * inverse.reshape(shape)
    if weight is not None:
        dx = dx * weight.reshape(shape)
    return tuple(value if wanted else None for value, wanted in
                 zip((dx, projected, total), output_mask))


def layer_norm(x, normalized_shape, weight, bias, eps):
    dims = tuple(range(x.ndim - len(normalized_shape), x.ndim))
    count = math.prod(normalized_shape)
    mean = reduce_sum(x, dims, True) / count
    centered = x - mean
    inverse = torch.rsqrt(reduce_sum(centered.square(), dims, True) / count + eps)
    output = centered * inverse
    if weight is not None:
        output = output * weight
    if bias is not None:
        output = output + bias
    return output, mean, inverse


def layer_norm_backward(gradient, x, normalized_shape, mean, inverse, weight, bias, output_mask):
    dims = tuple(range(x.ndim - len(normalized_shape), x.ndim))
    leading = tuple(range(x.ndim - len(normalized_shape)))
    normalized = (x - mean) * inverse
    weighted = gradient * weight if weight is not None else gradient
    count = math.prod(normalized_shape)
    dx = (weighted - reduce_sum(weighted, dims, True) / count
          - normalized * reduce_sum(weighted * normalized, dims, True) / count) * inverse
    dw = reduce_sum(gradient * normalized, leading) if leading else gradient * normalized
    db = reduce_sum(gradient, leading) if leading else gradient
    return tuple(value if wanted else None for value, wanted in zip((dx, dw, db), output_mask))


def softmax(x, dim):
    if not x.numel():
        return x.clone()
    exponent = torch.exp(x - x.amax(dim=dim, keepdim=True))
    return exponent / reduce_sum(exponent, (dim,), True)


def softmax_backward(gradient, output, dim, input_dtype):
    return (output * (gradient - reduce_sum(gradient * output, (dim,), True))).to(input_dtype)
