"""nvidia: explicitly reuse the installed vendor PyTorch implementation."""
from .base import MatrixOps
from . import torch_impl


class NvidiaMatrixOps(MatrixOps):
    dependencies = ("torch",)
    linear = staticmethod(torch_impl.linear)
    matmul = staticmethod(torch_impl.matmul)
    mm = staticmethod(torch_impl.mm)
    bmm = staticmethod(torch_impl.bmm)
    addmm = staticmethod(torch_impl.addmm)
    einsum = staticmethod(torch_impl.einsum)
    Linear = staticmethod(torch_impl.Linear)
