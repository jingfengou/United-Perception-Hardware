"""cpu: explicitly reuse the installed CPU PyTorch implementation."""
from .base import AttentionOps
from . import torch_impl


class CpuAttentionOps(AttentionOps):
    dependencies = ("torch",)
    MultiheadAttention = staticmethod(torch_impl.MultiheadAttention)
