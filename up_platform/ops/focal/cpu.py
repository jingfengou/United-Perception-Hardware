"""cpu focal: unimplemented contract template."""
from .base import FocalOps
from up_platform.contracts import UnsupportedOperation


class CpuFocalOps(FocalOps):
    dependencies = ()
    @staticmethod
    def sigmoid_focal_loss(preds, targets, weight_pos, gamma, alpha, num_classes, reduction):
        raise UnsupportedOperation("cpu", "unverified-template", "focal.sigmoid_focal_loss", "No native implementation validated")

    sigmoid_focal_loss.__func__.unsupported_reason = "Template; no native implementation"

    @staticmethod
    def softmax_focal_loss(preds, targets, weight_pos, gamma, alpha, num_classes):
        raise UnsupportedOperation("cpu", "unverified-template", "focal.softmax_focal_loss", "No native implementation validated")

    softmax_focal_loss.__func__.unsupported_reason = "Template; no native implementation"

    @staticmethod
    def cross_sigmoid_focal_loss(preds, targets, weight_pos, gamma, alpha, num_classes, gt_class_to_avoid, reduction):
        raise UnsupportedOperation("cpu", "unverified-template", "focal.cross_sigmoid_focal_loss", "No native implementation validated")

    cross_sigmoid_focal_loss.__func__.unsupported_reason = "Template; no native implementation"
