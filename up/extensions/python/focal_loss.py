from up_platform import get_platform
# Import from third library
import torch
from torch.autograd import Function

# Import from local


class SigmoidFocalLossFunction(Function):
    @staticmethod
    def apply(preds, targets, weight_pos, gamma, alpha, num_classes, reduction):
        return get_platform().category('focal').sigmoid_focal_loss(preds, targets, weight_pos, gamma, alpha, num_classes, reduction)


class SoftmaxFocalLossFunction(Function):
    @staticmethod
    def apply(preds, targets, weight_pos, gamma, alpha, num_classes):
        return get_platform().category('focal').softmax_focal_loss(preds, targets, weight_pos, gamma, alpha, num_classes)
