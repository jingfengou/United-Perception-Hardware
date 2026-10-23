from up_platform import get_platform
# Import from third library
import torch
from torch.autograd import Function

# Import from local


class CrossSigmoidFocalLossFunction(Function):
    @staticmethod
    def apply(preds, targets, weight_pos, gamma, alpha, num_classes, gt_class_to_avoid, reduction):
        return get_platform().category('focal').cross_sigmoid_focal_loss(preds, targets, weight_pos, gamma, alpha, num_classes, gt_class_to_avoid, reduction)
