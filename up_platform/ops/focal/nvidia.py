"""nvidia focal: own compiled binary, original UP algorithm."""
from .base import FocalOps
from up_platform.contracts import UnsupportedOperation


class NvidiaFocalOps(FocalOps):
    dependencies = ("up_platform.backends._nvidia_ops",)
    @staticmethod
    def sigmoid_focal_loss(preds, targets, weight_pos, gamma, alpha, num_classes, reduction):
        from ._focal_loss import SigmoidFocalLossFunction
        return SigmoidFocalLossFunction.apply(preds, targets, weight_pos, gamma, alpha, num_classes, reduction, "nvidia")

    @staticmethod
    def softmax_focal_loss(preds, targets, weight_pos, gamma, alpha, num_classes):
        from ._focal_loss import SoftmaxFocalLossFunction
        return SoftmaxFocalLossFunction.apply(preds, targets, weight_pos, gamma, alpha, num_classes, "nvidia")

    @staticmethod
    def cross_sigmoid_focal_loss(preds, targets, weight_pos, gamma, alpha, num_classes, gt_class_to_avoid, reduction):
        from ._cross_focal_loss import CrossSigmoidFocalLossFunction
        return CrossSigmoidFocalLossFunction.apply(preds, targets, weight_pos, gamma, alpha, num_classes, gt_class_to_avoid, reduction, "nvidia")
