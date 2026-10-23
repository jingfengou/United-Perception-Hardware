"""cambricon template: no operation in this category is claimed as supported."""
from .base import MatrixOps
from up_platform.contracts import UnsupportedOperation


class CambriconMatrixOps(MatrixOps):
    dependencies = ()
    @staticmethod
    def linear(input, weight, bias=None):
        raise UnsupportedOperation("cambricon", "unverified-template", "matrix.linear", "Implement and validate this contract before enabling the vendor")

    linear.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def matmul(input, other, *, out=None):
        raise UnsupportedOperation("cambricon", "unverified-template", "matrix.matmul", "Implement and validate this contract before enabling the vendor")

    matmul.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def mm(input, mat2, *, out=None):
        raise UnsupportedOperation("cambricon", "unverified-template", "matrix.mm", "Implement and validate this contract before enabling the vendor")

    mm.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def bmm(input, mat2, *, out=None):
        raise UnsupportedOperation("cambricon", "unverified-template", "matrix.bmm", "Implement and validate this contract before enabling the vendor")

    bmm.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
        raise UnsupportedOperation("cambricon", "unverified-template", "matrix.addmm", "Implement and validate this contract before enabling the vendor")

    addmm.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def einsum(equation, *operands):
        raise UnsupportedOperation("cambricon", "unverified-template", "matrix.einsum", "Implement and validate this contract before enabling the vendor")

    einsum.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def Linear(in_features, out_features, bias=True, device=None, dtype=None):
        raise UnsupportedOperation("cambricon", "unverified-template", "matrix.Linear", "Implement and validate this contract before enabling the vendor")

    Linear.__func__.unsupported_reason = "Vendor template; not implemented"
