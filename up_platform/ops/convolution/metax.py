"""metax: explicitly reuse the installed vendor PyTorch implementation."""
from .base import ConvolutionOps
from . import torch_impl


class MetaxConvolutionOps(ConvolutionOps):
    dependencies = ("torch",)
    conv2d = staticmethod(torch_impl.conv2d)
    conv_transpose2d = staticmethod(torch_impl.conv_transpose2d)
    Conv2d = staticmethod(torch_impl.Conv2d)
    ConvTranspose2d = staticmethod(torch_impl.ConvTranspose2d)
