"""Explicit nvidia device and operator selection."""
from importlib import import_module
import torch

from up_platform.base import PlatformBackend
from .torch_device import TorchDevice


class NvidiaDevice(TorchDevice):
    vendor = "nvidia"
    device_type = "cuda"

    def __init__(self, device=None):
        if getattr(torch._utils_internal, "USE_MACA", False) or "metax" in torch.__version__ or torch.version.hip:
            raise RuntimeError("nvidia requires an NVIDIA PyTorch build")
        super().__init__(device)


def build(config, purpose):
    device = NvidiaDevice(config.get("device"))
    operations = {
        "communication": getattr(import_module("up_platform.ops.communication.nvidia"), "NvidiaCommunicationOps")(),
        "sparse": getattr(import_module("up_platform.ops.sparse.nvidia"), "NvidiaSparseOps")(),
        "quantization": getattr(import_module("up_platform.ops.quantization.nvidia"), "NvidiaQuantizationOps")(),
        "pruning": getattr(import_module("up_platform.ops.pruning.nvidia"), "NvidiaPruningOps")(),
        "detection": getattr(import_module("up_platform.ops.detection.nvidia"), "NvidiaDetectionOps")(),
        "deformable": getattr(import_module("up_platform.ops.deformable.nvidia"), "NvidiaDeformableOps")(),
        "focal": getattr(import_module("up_platform.ops.focal.nvidia"), "NvidiaFocalOps")(),
        "pointcloud": getattr(import_module("up_platform.ops.pointcloud.nvidia"), "NvidiaPointcloudOps")(),
        "matrix": getattr(import_module("up_platform.ops.matrix.nvidia"), "NvidiaMatrixOps")(),
        "convolution": getattr(import_module("up_platform.ops.convolution.nvidia"), "NvidiaConvolutionOps")(),
        "normalization": getattr(import_module("up_platform.ops.normalization.nvidia"), "NvidiaNormalizationOps")(),
        "activation": getattr(import_module("up_platform.ops.activation.nvidia"), "NvidiaActivationOps")(),
        "pooling": getattr(import_module("up_platform.ops.pooling.nvidia"), "NvidiaPoolingOps")(),
        "loss": getattr(import_module("up_platform.ops.loss.nvidia"), "NvidiaLossOps")(),
        "attention": getattr(import_module("up_platform.ops.attention.nvidia"), "NvidiaAttentionOps")(),
    }
    execution = getattr(import_module("up_platform." + purpose + ".nvidia"), "Nvidia" + purpose.capitalize())(device)
    return PlatformBackend(device, purpose, operations, execution)
