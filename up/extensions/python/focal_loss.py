from up.accelerator import ops
# Import from third library
import torch
from torch.autograd import Function

# Import from local


class SigmoidFocalLossFunction:
    @staticmethod
    def apply(*args, **kwargs):
        return ops.sigmoid_focal_loss(*args, **kwargs)



class SoftmaxFocalLossFunction:
    @staticmethod
    def apply(*args, **kwargs):
        return ops.softmax_focal_loss(*args, **kwargs)
