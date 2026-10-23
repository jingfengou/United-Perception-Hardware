"""cambricon template: no operation in this category is claimed as supported."""
from .base import PoolingOps
from up_platform.contracts import UnsupportedOperation


class CambriconPoolingOps(PoolingOps):
    dependencies = ()
    @staticmethod
    def adaptive_avg_pool2d(input, output_size):
        raise UnsupportedOperation("cambricon", "unverified-template", "pooling.adaptive_avg_pool2d", "Implement and validate this contract before enabling the vendor")

    adaptive_avg_pool2d.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
        raise UnsupportedOperation("cambricon", "unverified-template", "pooling.max_pool2d", "Implement and validate this contract before enabling the vendor")

    max_pool2d.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
        raise UnsupportedOperation("cambricon", "unverified-template", "pooling.interpolate", "Implement and validate this contract before enabling the vendor")

    interpolate.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def pad(input, pad, mode='constant', value=None):
        raise UnsupportedOperation("cambricon", "unverified-template", "pooling.pad", "Implement and validate this contract before enabling the vendor")

    pad.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def AdaptiveAvgPool2d(output_size):
        raise UnsupportedOperation("cambricon", "unverified-template", "pooling.AdaptiveAvgPool2d", "Implement and validate this contract before enabling the vendor")

    AdaptiveAvgPool2d.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        raise UnsupportedOperation("cambricon", "unverified-template", "pooling.AvgPool2d", "Implement and validate this contract before enabling the vendor")

    AvgPool2d.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        raise UnsupportedOperation("cambricon", "unverified-template", "pooling.MaxPool2d", "Implement and validate this contract before enabling the vendor")

    MaxPool2d.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
        raise UnsupportedOperation("cambricon", "unverified-template", "pooling.Upsample", "Implement and validate this contract before enabling the vendor")

    Upsample.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def UpsamplingBilinear2d(size=None, scale_factor=None):
        raise UnsupportedOperation("cambricon", "unverified-template", "pooling.UpsamplingBilinear2d", "Implement and validate this contract before enabling the vendor")

    UpsamplingBilinear2d.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def ZeroPad2d(padding):
        raise UnsupportedOperation("cambricon", "unverified-template", "pooling.ZeroPad2d", "Implement and validate this contract before enabling the vendor")

    ZeroPad2d.__func__.unsupported_reason = "Vendor template; not implemented"
