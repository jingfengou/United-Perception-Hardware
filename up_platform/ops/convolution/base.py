"""Complete convolution operation contracts, independent of UP and vendor SDKs."""
from abc import ABC, abstractmethod
from up_platform.contracts import operation


class ConvolutionOps(ABC):
    @staticmethod
    @operation(source="torch.nn.functional.conv2d", output="NCHW output; preserve grouped/dilated convolution and original padding.",
               gradients="input, weight and optional bias", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        """NCHW output; preserve grouped/dilated convolution and original padding."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.conv_transpose2d", output="NCHW transpose-convolution output; preserve output_padding and original weight layout.",
               gradients="input, weight and optional bias", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        """NCHW transpose-convolution output; preserve output_padding and original weight layout."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.Conv2d", output="nn.Conv2d-compatible module; no wrapper prefix in state_dict; preserve RNG draws.",
               gradients="input and parameters", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        """nn.Conv2d-compatible module; no wrapper prefix in state_dict; preserve RNG draws."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.ConvTranspose2d", output="nn.ConvTranspose2d-compatible module with original state_dict and initialization.",
               gradients="input and parameters", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None):
        """nn.ConvTranspose2d-compatible module with original state_dict and initialization."""
        raise NotImplementedError
