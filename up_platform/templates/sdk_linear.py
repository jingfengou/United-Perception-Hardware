"""Example of routing a complete differentiable operation to a vendor SDK.

The supplied kernels must accept tensors on their existing device and return
tensors on that same device. This example does not provide a real GPU kernel.
Only first-order gradients are declared; higher-order derivatives are rejected.
"""
import torch
from torch.autograd.function import once_differentiable


def bind_sdk_linear(kernels):
    class SdkLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            ctx.save_for_backward(input, weight)
            ctx.has_bias = bias is not None
            return kernels.linear_forward(input, weight, bias)

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            input, weight = ctx.saved_tensors
            grad_input, grad_weight, grad_bias = kernels.linear_backward(
                grad_output, input, weight, ctx.has_bias, ctx.needs_input_grad)
            return grad_input, grad_weight, grad_bias

        @staticmethod
        def symbolic(graph, input, weight, bias):
            # An SDK-specific export must be supplied and tested, not guessed.
            raise RuntimeError('This example has no validated ONNX export adapter')

    def linear(input, weight, bias=None):
        return SdkLinear.apply(input, weight, bias)
    return linear
