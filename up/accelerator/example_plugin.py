"""Runnable CPU plugin illustrating shared implementation plus one override."""
from . import TorchAccelerator, register_backend, register_op
from .backends.portable import naive_nms as shared_nms
import torch


@register_backend("example_cpu")
class ExampleAccelerator(TorchAccelerator):
    name = "example_cpu"
    # A real plugin sets its own device_type and imports its SDK in __init__.
    device_type = "cpu"


@register_op("example_cpu", "naive_nms", backward=False, dtypes=(torch.float32,))
def vendor_nms(boxes, threshold):
    return shared_nms(boxes.contiguous(), threshold)
