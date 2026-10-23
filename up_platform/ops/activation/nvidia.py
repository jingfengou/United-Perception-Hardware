"""nvidia: explicitly reuse the installed vendor PyTorch implementation."""
from .base import ActivationOps
from . import torch_impl


class NvidiaActivationOps(ActivationOps):
    Hardswish = staticmethod(torch_impl.Hardswish)
    dependencies = ("torch",)
    relu = staticmethod(torch_impl.relu)
    relu6 = staticmethod(torch_impl.relu6)
    hardtanh = staticmethod(torch_impl.hardtanh)
    gelu = staticmethod(torch_impl.gelu)
    silu = staticmethod(torch_impl.silu)
    sigmoid = staticmethod(torch_impl.sigmoid)
    tanh = staticmethod(torch_impl.tanh)
    softmax = staticmethod(torch_impl.softmax)
    log_softmax = staticmethod(torch_impl.log_softmax)
    softplus = staticmethod(torch_impl.softplus)
    ReLU = staticmethod(torch_impl.ReLU)
    ReLU6 = staticmethod(torch_impl.ReLU6)
    GELU = staticmethod(torch_impl.GELU)
    SiLU = staticmethod(torch_impl.SiLU)
    SELU = staticmethod(torch_impl.SELU)
    LeakyReLU = staticmethod(torch_impl.LeakyReLU)
    Sigmoid = staticmethod(torch_impl.Sigmoid)
    Tanh = staticmethod(torch_impl.Tanh)
    Softmax = staticmethod(torch_impl.Softmax)
    Dropout = staticmethod(torch_impl.Dropout)
    Dropout2d = staticmethod(torch_impl.Dropout2d)
