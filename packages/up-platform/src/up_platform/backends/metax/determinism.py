"""Opt-in FP32 convolution weight gradients for the C500 repeatability issue."""
import threading

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from up_platform.errors import BackendConfigurationError, CapabilityError

PROFILE = "metax_deterministic_v1"


def convolution_weight_gradient(gradient, input, weight, stride, padding, dilation, groups):
    """Reduce each sample independently, then sum the batch in a fixed GPU tree."""
    batch = input.shape[0]
    columns = F.unfold(input, weight.shape[-2:], dilation=dilation, padding=padding, stride=stride)
    columns = columns.reshape(batch * groups, -1, columns.shape[-1])
    upstream = gradient.reshape(batch * groups, weight.shape[0] // groups, -1)
    partial = torch.bmm(upstream, columns.transpose(1, 2))
    return partial.reshape(batch, groups, weight.shape[0] // groups, -1).sum(dim=0).reshape_as(weight)


class DeterministicConvolution(TorchDispatchMode):
    """Keep native forward/input/bias gradients; replace only the weight reduction."""

    def __init__(self):
        super().__init__()
        self.owner_thread = threading.get_ident()
        self.counts = {}

    def check_thread(self):
        if threading.get_ident() != self.owner_thread:
            raise BackendConfigurationError(PROFILE + " must be reset on its owning thread")

    def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if operation is not torch.ops.aten.convolution_backward.default:
            return operation(*args, **kwargs)
        bound = dict(zip(("grad_output", "input", "weight", "bias_sizes", "stride", "padding", "dilation",
                          "transposed", "output_padding", "groups", "output_mask"), args))
        bound.update(kwargs)
        input, weight, gradient = bound["input"], bound["weight"], bound["grad_output"]
        if input.device.type != "cuda" or not bound["output_mask"][1]:
            return operation(*args, **kwargs)
        if (input.dtype != torch.float32 or weight.dtype != torch.float32 or gradient.dtype != torch.float32
                or input.ndim != 4 or weight.ndim != 4 or bound["transposed"] or input.shape[0] == 0):
            raise CapabilityError("metax/" + PROFILE + ": convolution weight gradient requires nonempty "
                                  "2D, non-transposed, dense GPU FP32 tensors; unsupported variant")
        native = dict(bound)
        mask = list(native["output_mask"])
        native["output_mask"] = [mask[0], False, mask[2]]
        dx, _, db = operation(**native)
        dw = convolution_weight_gradient(gradient, input, weight, bound["stride"], bound["padding"],
                                         bound["dilation"], bound["groups"])
        key = "aten.convolution_backward.weight"
        self.counts[key] = self.counts.get(key, 0) + 1
        return dx, dw, db
