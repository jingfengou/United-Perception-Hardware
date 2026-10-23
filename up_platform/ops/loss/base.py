"""Complete loss operation contracts, independent of UP and vendor SDKs."""
from abc import ABC, abstractmethod
from up_platform.contracts import operation


class LossOps(ABC):
    @staticmethod
    @operation(source="torch.nn.functional.binary_cross_entropy", output="Tensor scalar or elementwise by reduction; legacy size_average/reduce precedence preserved.",
               gradients="input and supported floating target", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction='mean'):
        """Tensor scalar or elementwise by reduction; legacy size_average/reduce precedence preserved."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.binary_cross_entropy_with_logits", output="Tensor scalar/elementwise; pos_weight broadcast and logits formulation unchanged.",
               gradients="input and supported floating target", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        """Tensor scalar/elementwise; pos_weight broadcast and logits formulation unchanged."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.cross_entropy", output="Scalar or unreduced loss, integer/probability target and ignore_index unchanged.",
               gradients="input and supported probability target", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0):
        """Scalar or unreduced loss, integer/probability target and ignore_index unchanged."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.kl_div", output="Loss with original reduction; mean and batchmean not interchangeable.",
               gradients="input and floating target", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def kl_div(input, target, size_average=None, reduce=None, reduction='mean', log_target=False):
        """Loss with original reduction; mean and batchmean not interchangeable."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.l1_loss", output="Scalar or elementwise L1 loss with original broadcast semantics.",
               gradients="input and target", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
        """Scalar or elementwise L1 loss with original broadcast semantics."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.CrossEntropyLoss", output="nn.CrossEntropyLoss; weight remains a registered buffer.",
               gradients="input", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0):
        """nn.CrossEntropyLoss; weight remains a registered buffer."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.BCEWithLogitsLoss", output="nn.BCEWithLogitsLoss; weight/pos_weight remain buffers.",
               gradients="input", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        """nn.BCEWithLogitsLoss; weight/pos_weight remain buffers."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.KLDivLoss", output="nn.KLDivLoss with original reduction.",
               gradients="input and floating target", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False):
        """nn.KLDivLoss with original reduction."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.L1Loss", output="nn.L1Loss with original reduction.",
               gradients="input and target", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def L1Loss(size_average=None, reduce=None, reduction='mean'):
        """nn.L1Loss with original reduction."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.CosineSimilarity", output="nn.CosineSimilarity module; preserve dim and eps.",
               gradients="both inputs", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def CosineSimilarity(dim=1, eps=1e-8):
        """nn.CosineSimilarity module; preserve dim and eps."""
        raise NotImplementedError
