"""ascend template: no operation in this category is claimed as supported."""
from .base import LossOps
from up_platform.contracts import UnsupportedOperation


class AscendLossOps(LossOps):
    dependencies = ()
    @staticmethod
    def binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction='mean'):
        raise UnsupportedOperation("ascend", "unverified-template", "loss.binary_cross_entropy", "Implement and validate this contract before enabling the vendor")

    binary_cross_entropy.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        raise UnsupportedOperation("ascend", "unverified-template", "loss.binary_cross_entropy_with_logits", "Implement and validate this contract before enabling the vendor")

    binary_cross_entropy_with_logits.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0):
        raise UnsupportedOperation("ascend", "unverified-template", "loss.cross_entropy", "Implement and validate this contract before enabling the vendor")

    cross_entropy.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def kl_div(input, target, size_average=None, reduce=None, reduction='mean', log_target=False):
        raise UnsupportedOperation("ascend", "unverified-template", "loss.kl_div", "Implement and validate this contract before enabling the vendor")

    kl_div.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
        raise UnsupportedOperation("ascend", "unverified-template", "loss.l1_loss", "Implement and validate this contract before enabling the vendor")

    l1_loss.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0):
        raise UnsupportedOperation("ascend", "unverified-template", "loss.CrossEntropyLoss", "Implement and validate this contract before enabling the vendor")

    CrossEntropyLoss.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        raise UnsupportedOperation("ascend", "unverified-template", "loss.BCEWithLogitsLoss", "Implement and validate this contract before enabling the vendor")

    BCEWithLogitsLoss.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False):
        raise UnsupportedOperation("ascend", "unverified-template", "loss.KLDivLoss", "Implement and validate this contract before enabling the vendor")

    KLDivLoss.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def L1Loss(size_average=None, reduce=None, reduction='mean'):
        raise UnsupportedOperation("ascend", "unverified-template", "loss.L1Loss", "Implement and validate this contract before enabling the vendor")

    L1Loss.__func__.unsupported_reason = "Vendor template; not implemented"

    @staticmethod
    def CosineSimilarity(dim=1, eps=1e-8):
        raise UnsupportedOperation("ascend", "unverified-template", "loss.CosineSimilarity", "Implement and validate this contract before enabling the vendor")

    CosineSimilarity.__func__.unsupported_reason = "Vendor template; not implemented"
