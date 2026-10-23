"""Complete pooling operation contracts, independent of UP and vendor SDKs."""
from abc import ABC, abstractmethod
from up_platform.contracts import operation


class PoolingOps(ABC):
    @staticmethod
    @operation(source="torch.nn.functional.adaptive_avg_pool2d", output="Tensor with requested H,W, same N,C.",
               gradients="input", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def adaptive_avg_pool2d(input, output_size):
        """Tensor with requested H,W, same N,C."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.max_pool2d", output="Tensor, or (Tensor, indices) when requested; indices retain upstream layout.",
               gradients="input only", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
        """Tensor, or (Tensor, indices) when requested; indices retain upstream layout."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.interpolate", output="Resized Tensor; all coordinate conventions and align_corners unchanged.",
               gradients="input", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
        """Resized Tensor; all coordinate conventions and align_corners unchanged."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.pad", output="Padded Tensor; pad order and mode unchanged.",
               gradients="input", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def pad(input, pad, mode='constant', value=None):
        """Padded Tensor; pad order and mode unchanged."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.AdaptiveAvgPool2d", output="nn.AdaptiveAvgPool2d module.",
               gradients="input", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def AdaptiveAvgPool2d(output_size):
        """nn.AdaptiveAvgPool2d module."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.AvgPool2d", output="nn.AvgPool2d module; divisor and padding semantics unchanged.",
               gradients="input", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        """nn.AvgPool2d module; divisor and padding semantics unchanged."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.MaxPool2d", output="nn.MaxPool2d module; return_indices and ceil_mode retain constructor order.",
               gradients="input", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        """nn.MaxPool2d module; return_indices and ceil_mode retain constructor order."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.Upsample", output="nn.Upsample module; original interpolation behavior.",
               gradients="input", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
        """nn.Upsample module; original interpolation behavior."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.UpsamplingBilinear2d", output="nn.UpsamplingBilinear2d module; original align_corners behavior.",
               gradients="input", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def UpsamplingBilinear2d(size=None, scale_factor=None):
        """nn.UpsamplingBilinear2d module; original align_corners behavior."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.ZeroPad2d", output="nn.ZeroPad2d module.",
               gradients="input", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def ZeroPad2d(padding):
        """nn.ZeroPad2d module."""
        raise NotImplementedError
