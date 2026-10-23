"""metax: explicitly reuse the installed vendor PyTorch implementation."""
from .base import AttentionOps
from . import torch_impl


class MetaxAttentionOps(AttentionOps):
    dependencies = ("torch",)
    MultiheadAttention = staticmethod(torch_impl.MultiheadAttention)
