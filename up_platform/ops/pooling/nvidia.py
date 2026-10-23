"""nvidia: explicitly reuse the installed vendor PyTorch implementation."""
from .base import PoolingOps
from . import torch_impl


class NvidiaPoolingOps(PoolingOps):
    dependencies = ("torch",)
    adaptive_avg_pool2d = staticmethod(torch_impl.adaptive_avg_pool2d)
    max_pool2d = staticmethod(torch_impl.max_pool2d)
    interpolate = staticmethod(torch_impl.interpolate)
    pad = staticmethod(torch_impl.pad)
    AdaptiveAvgPool2d = staticmethod(torch_impl.AdaptiveAvgPool2d)
    AvgPool2d = staticmethod(torch_impl.AvgPool2d)
    MaxPool2d = staticmethod(torch_impl.MaxPool2d)
    Upsample = staticmethod(torch_impl.Upsample)
    UpsamplingBilinear2d = staticmethod(torch_impl.UpsamplingBilinear2d)
    ZeroPad2d = staticmethod(torch_impl.ZeroPad2d)
