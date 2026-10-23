"""Relocated from UP 52aaeb9ed27a; arithmetic and autograd state retained."""
import logging
import numpy as np
import torch
from torch.autograd import Function
from up_platform.ops._native.loader import load_native
logger = logging.getLogger(__name__)


class DeformableConvFunction(Function):

    @staticmethod
    def symbolic(g, input, offset, weight, out_channels, kernel_size, stride, pad, dilations, groups, deformable_groups, vendor=None):
        assert input.type().sizes()[2] is not None, 'Input Error: Only 4D input Tensors Supported'
        assert input.type().sizes()[3] is not None, 'Input Error: Only 4D input Tensors Supported'
        return g.op('DeformConv', input, offset, weight, out_channels_i=out_channels, kernel_size_i=kernel_size, stride_i=stride, pad_i=pad, dilations_i=dilations, groups_i=groups, deformable_groups_i=deformable_groups)

    @staticmethod
    def forward(self, input, offset, weight, out_channels, kernel_size, stride, pad, dilations, groups, deformable_groups, vendor=None):
        self._deform_conv_v1 = load_native(vendor).deform_conv_v1
        if input is not None and input.dim() != 4:
            raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))
        self.save_for_backward(input, offset, weight)
        self.stride = stride
        self.padding = pad
        self.dilation = dilations
        self.groups = groups
        self.deformable_groups = deformable_groups
        output = input.new(*DeformableConvFunction._output_size(input, weight, self.stride, self.padding, self.dilation))
        self.bufs_ = [input.new(input.size(1) * weight.size(3) * weight.size(2), output.size(2), output.size(3)).zero_(), input.new(output.size(2), output.size(3)).fill_(1)]
        forward_fn = self._deform_conv_v1.forward_cuda
        if not input.is_cuda:
            logger.warning('---CPU version of DEFORMABLE CONV V1 is a dummpy function, which is used to support tocaffe')
            forward_fn = self._deform_conv_v1.forward_cpu
        forward_fn(input, weight, offset, output, self.bufs_[0], self.bufs_[1], weight.size(2), weight.size(3), self.stride[0], self.stride[1], self.padding[0], self.padding[1], self.dilation[0], self.dilation[1], self.groups, self.deformable_groups)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, offset, weight = self.saved_tensors
        grad_input = None
        grad_offset = None
        grad_weight = None
        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            grad_output = grad_output.contiguous()
            if self.needs_input_grad[0] or self.needs_input_grad[1]:
                grad_input = input.new(*input.size()).zero_()
                grad_offset = offset.new(*offset.size()).zero_()
                self._deform_conv_v1.backward_input_cuda(input, offset, grad_output, grad_input, grad_offset, weight, self.bufs_[0], weight.size(2), weight.size(3), self.stride[0], self.stride[1], self.padding[0], self.padding[1], self.dilation[0], self.dilation[1], self.groups, self.deformable_groups)
            if self.needs_input_grad[2]:
                grad_weight = weight.new(*weight.size()).zero_()
                self._deform_conv_v1.backward_parameters_cuda(input, offset, grad_output, grad_weight, self.bufs_[0], self.bufs_[1], weight.size(2), weight.size(3), self.stride[0], self.stride[1], self.padding[0], self.padding[1], self.dilation[0], self.dilation[1], self.groups, self.deformable_groups, 1)
        return (grad_input, grad_offset, grad_weight, None, None, None, None, None, None, None, None)

    @staticmethod
    def _output_size(input, weight, stride, padding, dilation):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            _in_size = input.size(d + 2)
            _pad = padding[d]
            _kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            _stride = stride[d]
            output_size += ((_in_size + 2 * _pad - _kernel) // _stride + 1,)
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError('convolution input is too small (output would be {})'.format('x'.join(map(str, output_size))))
        return output_size
