from up_platform import get_platform
# Standard Library
import math

# Import from third library
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair

# Import from up
from up.utils.general.log_helper import default_logger as logger

# Import from local


class DeformableConvFunction(Function):
    @staticmethod
    def apply(input, offset, weight, out_channels, kernel_size, stride, pad, dilations, groups, deformable_groups):
        return get_platform().category('deformable').deform_conv(input, offset, weight, out_channels, kernel_size, stride, pad, dilations, groups, deformable_groups)


class DeformableConv(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 num_deformable_groups=1):
        super(DeformableConv, self).__init__()

        # logger.warning('warning! DeformableConv will be deprecated in the near future, '
        #                'plase use DeformConv2d instead, which is an unified module with '
        #                'the same interface with torch.nn.Conv2d')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.num_deformable_groups = num_deformable_groups

        assert in_channels % groups == 0
        assert out_channels % groups == 0
        assert (in_channels // groups % num_deformable_groups == 0)
        assert (out_channels // groups % num_deformable_groups == 0)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, offset):
        # logger.info(self.stride)
        return DeformableConvFunction.apply(
            input, offset, self.weight, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation,
            self.groups, self.num_deformable_groups)

    def __repr__(self):
        s = ('{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, num_deformable_groups={num_deformable_groups}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
