"""ascend template: no operation in this category is claimed as supported."""
from .base import NormalizationOps
from up_platform.contracts import UnsupportedOperation


class AscendNormalizationOps(NormalizationOps):
    @staticmethod
    def batch_norm_module_forward(module, input):
        raise UnsupportedOperation("ascend", "common", "normalization.batch_norm_module_forward", "Unimplemented vendor template")
    batch_norm_module_forward.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def sync_batch_norm_forward(module, input):
        raise UnsupportedOperation("ascend", "common", "normalization.sync_batch_norm_forward", "Unimplemented vendor template")
    sync_batch_norm_forward.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def sync_batch_norm(input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        raise UnsupportedOperation("ascend", "common", "normalization.sync_batch_norm", "Unimplemented vendor template")
    sync_batch_norm.__func__.unsupported_reason = "Unimplemented vendor template"

    dependencies = ()
    @staticmethod
    def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
        raise UnsupportedOperation("ascend", "unverified-template", "normalization.batch_norm", "Implement and validate this contract before enabling the vendor")

    batch_norm.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        raise UnsupportedOperation("ascend", "unverified-template", "normalization.layer_norm", "Implement and validate this contract before enabling the vendor")

    layer_norm.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
        raise UnsupportedOperation("ascend", "unverified-template", "normalization.group_norm", "Implement and validate this contract before enabling the vendor")

    group_norm.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def normalize(input, p=2.0, dim=1, eps=1e-12, out=None):
        raise UnsupportedOperation("ascend", "unverified-template", "normalization.normalize", "Implement and validate this contract before enabling the vendor")

    normalize.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        raise UnsupportedOperation("ascend", "unverified-template", "normalization.BatchNorm1d", "Implement and validate this contract before enabling the vendor")

    BatchNorm1d.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        raise UnsupportedOperation("ascend", "unverified-template", "normalization.BatchNorm2d", "Implement and validate this contract before enabling the vendor")

    BatchNorm2d.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None):
        raise UnsupportedOperation("ascend", "unverified-template", "normalization.LayerNorm", "Implement and validate this contract before enabling the vendor")

    LayerNorm.__func__.unsupported_reason = "Vendor template; not implemented"
