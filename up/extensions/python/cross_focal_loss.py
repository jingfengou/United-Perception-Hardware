from up.accelerator import ops
# Import from third library
import torch
from torch.autograd import Function

# Import from local


class CrossSigmoidFocalLossFunction:
    @staticmethod
    def apply(*args, **kwargs):
        return ops.cross_focal_loss(*args, **kwargs)
