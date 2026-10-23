"""ascend: explicit sparse bindings; registration is not validation."""
from .base import SparseOps
from up_platform.contracts import UnsupportedOperation


class AscendSparseOps(SparseOps):
    @staticmethod
    def voxelize_cpu(generator, points):
        raise UnsupportedOperation("ascend", "common", "sparse.voxelize_cpu", "Unimplemented vendor template")
    voxelize_cpu.__func__.unsupported_reason = "Unimplemented vendor template"


    @staticmethod
    def SubMConv3d(in_channels, out_channels, kernel_size, **kwargs):
        raise UnsupportedOperation("ascend", "common", "sparse.SubMConv3d", "Vendor template: implementation and hardware validation required")
    SubMConv3d.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def SparseConv3d(in_channels, out_channels, kernel_size, **kwargs):
        raise UnsupportedOperation("ascend", "common", "sparse.SparseConv3d", "Vendor template: implementation and hardware validation required")
    SparseConv3d.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key, **kwargs):
        raise UnsupportedOperation("ascend", "common", "sparse.SparseInverseConv3d", "Vendor template: implementation and hardware validation required")
    SparseInverseConv3d.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def SparseSequential(*modules):
        raise UnsupportedOperation("ascend", "common", "sparse.SparseSequential", "Vendor template: implementation and hardware validation required")
    SparseSequential.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def SparseConvTensor(features, indices, spatial_shape, batch_size, **kwargs):
        raise UnsupportedOperation("ascend", "common", "sparse.SparseConvTensor", "Vendor template: implementation and hardware validation required")
    SparseConvTensor.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def replace_feature(tensor, features):
        raise UnsupportedOperation("ascend", "common", "sparse.replace_feature", "Vendor template: implementation and hardware validation required")
    replace_feature.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def features(tensor):
        raise UnsupportedOperation("ascend", "common", "sparse.features", "Vendor template: implementation and hardware validation required")
    features.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def dense(tensor, channels_first=True):
        raise UnsupportedOperation("ascend", "common", "sparse.dense", "Vendor template: implementation and hardware validation required")
    dense.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def is_sparse_convolution(module):
        raise UnsupportedOperation("ascend", "common", "sparse.is_sparse_convolution", "Vendor template: implementation and hardware validation required")
    is_sparse_convolution.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def module_instance(module_type):
        raise UnsupportedOperation("ascend", "common", "sparse.module_instance", "Vendor template: implementation and hardware validation required")
    module_instance.__func__.unsupported_reason = "Unimplemented vendor template"

