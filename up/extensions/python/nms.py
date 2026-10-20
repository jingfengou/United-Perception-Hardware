from up.accelerator import ops
# Import from third library
import numpy as np
import torch
from up.flags import ALIGNED_FLAG


def naive_nms(*args, **kwargs):
    return ops.naive_nms(*args, **kwargs)


def softer_nms(*args, **kwargs):
    return ops.softer_nms(*args, **kwargs)
