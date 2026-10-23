"""ascend template: no operation in this category is claimed as supported."""
from .base import ActivationOps
from up_platform.contracts import UnsupportedOperation


class AscendActivationOps(ActivationOps):
    @staticmethod
    def Hardswish(inplace=False):
        raise UnsupportedOperation("ascend", "common", "activation.Hardswish", "Unimplemented vendor template")
    Hardswish.__func__.unsupported_reason = "Unimplemented vendor template"

    dependencies = ()
    @staticmethod
    def relu(input, inplace=False):
        raise UnsupportedOperation("ascend", "unverified-template", "activation.relu", "Implement and validate this contract before enabling the vendor")

    relu.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def relu6(input, inplace=False):
        raise UnsupportedOperation("ascend", "unverified-template", "activation.relu6", "Implement and validate this contract before enabling the vendor")

    relu6.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
        raise UnsupportedOperation("ascend", "unverified-template", "activation.hardtanh", "Implement and validate this contract before enabling the vendor")

    hardtanh.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def gelu(input, approximate='none'):
        raise UnsupportedOperation("ascend", "unverified-template", "activation.gelu", "Implement and validate this contract before enabling the vendor")

    gelu.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def silu(input, inplace=False):
        raise UnsupportedOperation("ascend", "unverified-template", "activation.silu", "Implement and validate this contract before enabling the vendor")

    silu.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def sigmoid(input):
        raise UnsupportedOperation("ascend", "unverified-template", "activation.sigmoid", "Implement and validate this contract before enabling the vendor")

    sigmoid.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def tanh(input):
        raise UnsupportedOperation("ascend", "unverified-template", "activation.tanh", "Implement and validate this contract before enabling the vendor")

    tanh.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def softmax(input, dim=None, _stacklevel=3, dtype=None):
        raise UnsupportedOperation("ascend", "unverified-template", "activation.softmax", "Implement and validate this contract before enabling the vendor")

    softmax.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
        raise UnsupportedOperation("ascend", "unverified-template", "activation.log_softmax", "Implement and validate this contract before enabling the vendor")

    log_softmax.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def softplus(input, beta=1, threshold=20):
        raise UnsupportedOperation("ascend", "unverified-template", "activation.softplus", "Implement and validate this contract before enabling the vendor")

    softplus.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def ReLU(inplace=False):
        raise UnsupportedOperation("ascend", "unverified-template", "activation.ReLU", "Implement and validate this contract before enabling the vendor")

    ReLU.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def ReLU6(inplace=False):
        raise UnsupportedOperation("ascend", "unverified-template", "activation.ReLU6", "Implement and validate this contract before enabling the vendor")

    ReLU6.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def GELU(approximate='none'):
        raise UnsupportedOperation("ascend", "unverified-template", "activation.GELU", "Implement and validate this contract before enabling the vendor")

    GELU.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def SiLU(inplace=False):
        raise UnsupportedOperation("ascend", "unverified-template", "activation.SiLU", "Implement and validate this contract before enabling the vendor")

    SiLU.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def SELU(inplace=False):
        raise UnsupportedOperation("ascend", "unverified-template", "activation.SELU", "Implement and validate this contract before enabling the vendor")

    SELU.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def LeakyReLU(negative_slope=0.01, inplace=False):
        raise UnsupportedOperation("ascend", "unverified-template", "activation.LeakyReLU", "Implement and validate this contract before enabling the vendor")

    LeakyReLU.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def Sigmoid():
        raise UnsupportedOperation("ascend", "unverified-template", "activation.Sigmoid", "Implement and validate this contract before enabling the vendor")

    Sigmoid.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def Tanh():
        raise UnsupportedOperation("ascend", "unverified-template", "activation.Tanh", "Implement and validate this contract before enabling the vendor")

    Tanh.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def Softmax(dim=None):
        raise UnsupportedOperation("ascend", "unverified-template", "activation.Softmax", "Implement and validate this contract before enabling the vendor")

    Softmax.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def Dropout(p=0.5, inplace=False):
        raise UnsupportedOperation("ascend", "unverified-template", "activation.Dropout", "Implement and validate this contract before enabling the vendor")

    Dropout.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def Dropout2d(p=0.5, inplace=False):
        raise UnsupportedOperation("ascend", "unverified-template", "activation.Dropout2d", "Implement and validate this contract before enabling the vendor")

    Dropout2d.__func__.unsupported_reason = "Vendor template; not implemented"
