"""metax: explicitly reuse the installed vendor PyTorch implementation."""
from .base import NormalizationOps
from . import torch_impl


class MetaxNormalizationOps(NormalizationOps):
    batch_norm_module_forward = staticmethod(torch_impl.batch_norm_module_forward)
    sync_batch_norm_forward = staticmethod(torch_impl.sync_batch_norm_forward)
    sync_batch_norm = staticmethod(torch_impl.sync_batch_norm)
    dependencies = ("torch",)
    batch_norm = staticmethod(torch_impl.batch_norm)
    layer_norm = staticmethod(torch_impl.layer_norm)
    group_norm = staticmethod(torch_impl.group_norm)
    normalize = staticmethod(torch_impl.normalize)
    BatchNorm1d = staticmethod(torch_impl.BatchNorm1d)
    BatchNorm2d = staticmethod(torch_impl.BatchNorm2d)
    LayerNorm = staticmethod(torch_impl.LayerNorm)
