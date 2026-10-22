"""Complete shared 2D convolution routing, including the autograd operation."""
import torch
import torch.nn.functional as F

from up_platform.errors import CapabilityError
from .portable_math import matrix, reduce_sum


def columns(input, weight, stride, padding, dilation, groups):
    if input.ndim != 4 or weight.ndim != 4:
        raise CapabilityError("portable_v1 shared convolution requires 2D inputs")
    if input.shape[1] != weight.shape[1] * groups or weight.shape[0] % groups:
        raise ValueError("Incompatible convolution groups")
    value = F.unfold(input, weight.shape[-2:], dilation=dilation, padding=padding, stride=stride)
    return value.reshape(input.shape[0] * groups, -1, value.shape[-1])


def grouped_weight(weight, batch, groups):
    return weight.reshape(1, groups, weight.shape[0] // groups, -1).expand(
        batch, -1, -1, -1).reshape(batch * groups, weight.shape[0] // groups, -1)


def convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    if transposed:
        raise CapabilityError("portable_v1 shared convolution does not implement transpose convolution")
    batch = input.shape[0]
    data = columns(input, weight, stride, padding, dilation, groups)
    product = matrix(grouped_weight(weight, batch, groups), data)
    height, width = [(input.shape[d + 2] + 2 * padding[d] - dilation[d] * (weight.shape[d + 2] - 1) - 1)
                     // stride[d] + 1 for d in range(2)]
    output = product.reshape(batch, weight.shape[0], height, width)
    return output if bias is None else output + bias.reshape(1, -1, 1, 1)


def convolution_backward(gradient, input, weight, bias_sizes, stride, padding, dilation,
                         transposed, output_padding, groups, output_mask):
    if transposed:
        raise CapabilityError("portable_v1 shared convolution backward does not implement transpose convolution")
    batch = input.shape[0]
    group_gradient = gradient.reshape(batch * groups, weight.shape[0] // groups, -1)
    dx = dw = db = None
    if output_mask[0]:
        transformed = matrix(grouped_weight(weight, batch, groups).transpose(-1, -2), group_gradient)
        transformed = transformed.reshape(batch, -1, group_gradient.shape[-1])
        dx = F.fold(transformed, input.shape[-2:], weight.shape[-2:],
                    dilation=dilation, padding=padding, stride=stride)
    if output_mask[1]:
        data = columns(input, weight, stride, padding, dilation, groups)
        partial = matrix(group_gradient, data.transpose(-1, -2))
        dw = reduce_sum(partial.reshape(batch, groups, weight.shape[0] // groups, -1), (0,)).reshape_as(weight)
    if output_mask[2]:
        db = reduce_sum(gradient, (0, 2, 3))
    return dx, dw, db
