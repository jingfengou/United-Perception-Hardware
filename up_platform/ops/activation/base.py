"""Complete activation operation contracts, independent of UP and vendor SDKs."""
from abc import ABC, abstractmethod
from up_platform.contracts import operation


class ActivationOps(ABC):
    @staticmethod
    @operation(source="torch.nn.Hardswish", output="Original PyTorch result.", gradients="Original input and parameter gradients.", state="Original module and autograd state updates.")
    @abstractmethod
    def Hardswish(inplace=False):
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.relu", output="Same-shape Tensor; preserve in-place alias/version-counter behavior.",
               gradients="input, subject to upstream in-place restrictions", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def relu(input, inplace=False):
        """Same-shape Tensor; preserve in-place alias/version-counter behavior."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.relu6", output="Same-shape Tensor clipped to [0,6]; preserve in-place behavior.",
               gradients="input", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def relu6(input, inplace=False):
        """Same-shape Tensor clipped to [0,6]; preserve in-place behavior."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.hardtanh", output="Same-shape Tensor clamped to min_val/max_val.",
               gradients="input", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
        """Same-shape Tensor clamped to min_val/max_val."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.gelu", output="Same-shape Tensor; do not change approximation mode.",
               gradients="input", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def gelu(input, approximate='none'):
        """Same-shape Tensor; do not change approximation mode."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.silu", output="Same-shape Tensor input*sigmoid(input).",
               gradients="input", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def silu(input, inplace=False):
        """Same-shape Tensor input*sigmoid(input)."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.sigmoid", output="Same-shape Tensor with original Torch sigmoid semantics.",
               gradients="input", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def sigmoid(input):
        """Same-shape Tensor with original Torch sigmoid semantics."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.tanh", output="Same-shape Tensor with original Torch tanh semantics.",
               gradients="input", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def tanh(input):
        """Same-shape Tensor with original Torch tanh semantics."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.softmax", output="Same-shape Tensor normalized along dim; preserve optional dtype conversion.",
               gradients="input", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def softmax(input, dim=None, _stacklevel=3, dtype=None):
        """Same-shape Tensor normalized along dim; preserve optional dtype conversion."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.log_softmax", output="Same-shape Tensor, numerically stable log-softmax, unchanged dim/dtype.",
               gradients="input", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
        """Same-shape Tensor, numerically stable log-softmax, unchanged dim/dtype."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.softplus", output="Same-shape Tensor with original threshold branch.",
               gradients="input", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def softplus(input, beta=1, threshold=20):
        """Same-shape Tensor with original threshold branch."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.ReLU", output="nn.ReLU module preserving in-place behavior.",
               gradients="input", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def ReLU(inplace=False):
        """nn.ReLU module preserving in-place behavior."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.ReLU6", output="nn.ReLU6 module.",
               gradients="input", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def ReLU6(inplace=False):
        """nn.ReLU6 module."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.GELU", output="nn.GELU module, approximation unchanged.",
               gradients="input", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def GELU(approximate='none'):
        """nn.GELU module, approximation unchanged."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.SiLU", output="nn.SiLU module.",
               gradients="input", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def SiLU(inplace=False):
        """nn.SiLU module."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.SELU", output="nn.SELU module.",
               gradients="input", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def SELU(inplace=False):
        """nn.SELU module."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.LeakyReLU", output="nn.LeakyReLU module, slope and alias semantics unchanged.",
               gradients="input", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def LeakyReLU(negative_slope=0.01, inplace=False):
        """nn.LeakyReLU module, slope and alias semantics unchanged."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.Sigmoid", output="nn.Sigmoid module.",
               gradients="input", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def Sigmoid():
        """nn.Sigmoid module."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.Tanh", output="nn.Tanh module.",
               gradients="input", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def Tanh():
        """nn.Tanh module."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.Softmax", output="nn.Softmax module, preserve dim.",
               gradients="input", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def Softmax(dim=None):
        """nn.Softmax module, preserve dim."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.Dropout", output="nn.Dropout module; training flag, RNG consumption and mask scaling unchanged.",
               gradients="input", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def Dropout(p=0.5, inplace=False):
        """nn.Dropout module; training flag, RNG consumption and mask scaling unchanged."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.Dropout2d", output="nn.Dropout2d module; channel mask, training flag and RNG unchanged.",
               gradients="input", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def Dropout2d(p=0.5, inplace=False):
        """nn.Dropout2d module; channel mask, training flag and RNG unchanged."""
        raise NotImplementedError
