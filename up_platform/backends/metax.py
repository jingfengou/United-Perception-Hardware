"""Explicit metax device and operator selection."""
from importlib import import_module
import torch

from up_platform.base import PlatformBackend
from .torch_device import TorchDevice


class MetaxDevice(TorchDevice):
    vendor = "metax"
    device_type = "cuda"

    def __init__(self, device=None):
        if not (getattr(torch._utils_internal, "USE_MACA", False) or "metax" in torch.__version__):
            raise RuntimeError("metax requires a matching MetaX PyTorch build, not an NVIDIA wheel")
        super().__init__(device)


def build(config, purpose):
    device = MetaxDevice(config.get("device"))
    operations = {
        "communication": getattr(import_module("up_platform.ops.communication.metax"), "MetaxCommunicationOps")(),
        "sparse": getattr(import_module("up_platform.ops.sparse.metax"), "MetaxSparseOps")(),
        "quantization": getattr(import_module("up_platform.ops.quantization.metax"), "MetaxQuantizationOps")(),
        "pruning": getattr(import_module("up_platform.ops.pruning.metax"), "MetaxPruningOps")(),
        "detection": getattr(import_module("up_platform.ops.detection.metax"), "MetaxDetectionOps")(),
        "deformable": getattr(import_module("up_platform.ops.deformable.metax"), "MetaxDeformableOps")(),
        "focal": getattr(import_module("up_platform.ops.focal.metax"), "MetaxFocalOps")(),
        "pointcloud": getattr(import_module("up_platform.ops.pointcloud.metax"), "MetaxPointcloudOps")(),
        "matrix": getattr(import_module("up_platform.ops.matrix.metax"), "MetaxMatrixOps")(),
        "convolution": getattr(import_module("up_platform.ops.convolution.metax"), "MetaxConvolutionOps")(),
        "normalization": getattr(import_module("up_platform.ops.normalization.metax"), "MetaxNormalizationOps")(),
        "activation": getattr(import_module("up_platform.ops.activation.metax"), "MetaxActivationOps")(),
        "pooling": getattr(import_module("up_platform.ops.pooling.metax"), "MetaxPoolingOps")(),
        "loss": getattr(import_module("up_platform.ops.loss.metax"), "MetaxLossOps")(),
        "attention": getattr(import_module("up_platform.ops.attention.metax"), "MetaxAttentionOps")(),
    }
    execution = getattr(import_module("up_platform." + purpose + ".metax"), "Metax" + purpose.capitalize())(device)
    return PlatformBackend(device, purpose, operations, execution)
