"""cambricon template: no operation in this category is claimed as supported."""
from .base import ConvolutionOps
from up_platform.contracts import UnsupportedOperation


class CambriconConvolutionOps(ConvolutionOps):
    dependencies = ()
    @staticmethod
    def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        raise UnsupportedOperation("cambricon", "unverified-template", "convolution.conv2d", "Implement and validate this contract before enabling the vendor")

    conv2d.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        raise UnsupportedOperation("cambricon", "unverified-template", "convolution.conv_transpose2d", "Implement and validate this contract before enabling the vendor")

    conv_transpose2d.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        raise UnsupportedOperation("cambricon", "unverified-template", "convolution.Conv2d", "Implement and validate this contract before enabling the vendor")

    Conv2d.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None):
        raise UnsupportedOperation("cambricon", "unverified-template", "convolution.ConvTranspose2d", "Implement and validate this contract before enabling the vendor")

    ConvTranspose2d.__func__.unsupported_reason = "Vendor template; not implemented"
