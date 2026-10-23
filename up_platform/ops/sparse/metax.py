"""metax: explicit sparse bindings; registration is not validation."""
from .base import SparseOps
from . import torch_impl


class MetaxSparseOps(SparseOps):
    voxelize_cpu = staticmethod(torch_impl.voxelize_cpu)
    dependencies = ("spconv",)
    SubMConv3d = staticmethod(torch_impl.SubMConv3d)
    SparseConv3d = staticmethod(torch_impl.SparseConv3d)
    SparseInverseConv3d = staticmethod(torch_impl.SparseInverseConv3d)
    SparseSequential = staticmethod(torch_impl.SparseSequential)
    SparseConvTensor = staticmethod(torch_impl.SparseConvTensor)
    replace_feature = staticmethod(torch_impl.replace_feature)
    features = staticmethod(torch_impl.features)
    @staticmethod
    def dense(tensor, channels_first=True):
        # MetaX spconv 2.1 returns N...C and has no channels_first keyword.
        result = tensor.dense()
        if channels_first:
            result = result.permute(0, result.ndim - 1, *range(1, result.ndim - 1)).contiguous()
        return result
    is_sparse_convolution = staticmethod(torch_impl.is_sparse_convolution)
    module_instance = staticmethod(torch_impl.module_instance)

