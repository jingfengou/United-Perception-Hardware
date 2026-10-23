"""Complete matrix operation contracts, independent of UP and vendor SDKs."""
from abc import ABC, abstractmethod
from up_platform.contracts import operation


class MatrixOps(ABC):
    @staticmethod
    @operation(source="torch.nn.functional.linear", output="Tensor (*, out_features); weight=(out_features, in_features); optional bias=(out_features,).",
               gradients="input, weight and optional bias", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def linear(input, weight, bias=None):
        """Tensor (*, out_features); weight=(out_features, in_features); optional bias=(out_features,)."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.matmul", output="Tensor with PyTorch vector/matrix/batched broadcasting; out follows PyTorch autograd restrictions.",
               gradients="input and other unless out is used", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def matmul(input, other, *, out=None):
        """Tensor with PyTorch vector/matrix/batched broadcasting; out follows PyTorch autograd restrictions."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.mm", output="Tensor (N,P) from (N,M) and (M,P); no broadcasting.",
               gradients="input and mat2 unless out is used", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def mm(input, mat2, *, out=None):
        """Tensor (N,P) from (N,M) and (M,P); no broadcasting."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.bmm", output="Tensor (B,N,P) from (B,N,M) and (B,M,P); no broadcasting.",
               gradients="input and mat2 unless out is used", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def bmm(input, mat2, *, out=None):
        """Tensor (B,N,P) from (B,N,M) and (B,M,P); no broadcasting."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.addmm", output="Tensor beta*input + alpha*(mat1@mat2), following upstream accumulation and promotion.",
               gradients="floating inputs unless out is used", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
        """Tensor beta*input + alpha*(mat1@mat2), following upstream accumulation and promotion."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.einsum", output="Tensor according to the original string equation; operand order is unchanged.",
               gradients="floating operands", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def einsum(equation, *operands):
        """Tensor according to the original string equation; operand order is unchanged."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.Linear", output="nn.Linear-compatible module with original weight/bias keys, shapes and initialization.",
               gradients="input and parameters", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def Linear(in_features, out_features, bias=True, device=None, dtype=None):
        """nn.Linear-compatible module with original weight/bias keys, shapes and initialization."""
        raise NotImplementedError
