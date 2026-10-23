"""Relocated from UP 52aaeb9ed27a; arithmetic and autograd state retained."""
import logging
import numpy as np
import torch
from torch.autograd import Function
from up_platform.ops._native.loader import load_native
logger = logging.getLogger(__name__)


class CrossSigmoidFocalLossFunction(Function):

    @staticmethod
    def forward(self, preds, targets, weight_pos, gamma, alpha, num_classes, gt_class_to_avoid, reduction, vendor=None):
        """
        Arguments:
            preds: [batch * h * w * num_anchors, num_classes]
            targets: [batch * h * w * num_anchors]
            weight_pos: Scalar Tensor of normalizer
        """
        self._cross_focal_loss = load_native(vendor).cross_focal_loss
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        self.gt_class_to_avoid = gt_class_to_avoid
        preds_size = preds.size()
        targets_size = targets.size()
        assert preds_size[0] == targets_size[0]
        assert preds_size[1] == self.num_classes
        losses = preds.new(preds_size[0], preds_size[1]).zero_()
        N = preds_size[0] * preds_size[1]
        assert losses.is_contiguous()
        assert preds.is_contiguous()
        assert targets.is_contiguous()
        assert preds.is_cuda and targets.is_cuda
        self._cross_focal_loss.cross_sigmoid_forward_cuda(N, preds, targets, weight_pos.cpu().item(), self.gamma, self.alpha, self.num_classes, losses, self.gt_class_to_avoid)
        self.save_for_backward(preds, targets, weight_pos, gt_class_to_avoid)
        return torch.cuda.FloatTensor([losses.sum()])
        if reduction == 'none':
            return torch.cuda.FloatTensor(losses)
        else:
            return torch.cuda.FloatTensor([losses.sum()])

    @staticmethod
    def backward(self, grad_output):
        preds, targets, weight_pos, gt_class_to_avoid = self.saved_tensors
        preds_size = preds.size()
        grad_input = preds.new(preds_size[0], preds_size[1]).zero_()
        N = preds_size[0] * preds_size[1]
        assert preds.is_contiguous()
        assert targets.is_contiguous()
        assert grad_input.is_contiguous()
        assert preds.is_cuda and targets.is_cuda and grad_input.is_cuda
        self._cross_focal_loss.cross_sigmoid_backward_cuda(N, preds, targets, grad_input, weight_pos.cpu().item(), self.gamma, self.alpha, self.num_classes, gt_class_to_avoid)
        grad_input = grad_input * grad_output
        return (grad_input, None, None, None, None, None, None, None, None)
