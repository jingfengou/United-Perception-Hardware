"""cpu: explicitly reuse the installed CPU PyTorch implementation."""
from .base import ConvolutionOps
from . import torch_impl


class CpuConvolutionOps(ConvolutionOps):
    dependencies = ("torch",)
    conv2d = staticmethod(torch_impl.conv2d)
    conv_transpose2d = staticmethod(torch_impl.conv_transpose2d)
    Conv2d = staticmethod(torch_impl.Conv2d)
    ConvTranspose2d = staticmethod(torch_impl.ConvTranspose2d)
