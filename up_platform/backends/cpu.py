"""Explicit cpu device and operator selection."""
from importlib import import_module
import torch

from up_platform.base import PlatformBackend
from .torch_device import TorchDevice


class CpuDevice(TorchDevice):
    vendor = "cpu"
    device_type = "cpu"

    def __init__(self, device=None):
        super().__init__(device)


def build(config, purpose):
    device = CpuDevice(config.get("device"))
    operations = {
        "communication": getattr(import_module("up_platform.ops.communication.cpu"), "CpuCommunicationOps")(),
        "sparse": getattr(import_module("up_platform.ops.sparse.cpu"), "CpuSparseOps")(),
        "quantization": getattr(import_module("up_platform.ops.quantization.cpu"), "CpuQuantizationOps")(),
        "pruning": getattr(import_module("up_platform.ops.pruning.cpu"), "CpuPruningOps")(),
        "detection": getattr(import_module("up_platform.ops.detection.cpu"), "CpuDetectionOps")(),
        "deformable": getattr(import_module("up_platform.ops.deformable.cpu"), "CpuDeformableOps")(),
        "focal": getattr(import_module("up_platform.ops.focal.cpu"), "CpuFocalOps")(),
        "pointcloud": getattr(import_module("up_platform.ops.pointcloud.cpu"), "CpuPointcloudOps")(),
        "matrix": getattr(import_module("up_platform.ops.matrix.cpu"), "CpuMatrixOps")(),
        "convolution": getattr(import_module("up_platform.ops.convolution.cpu"), "CpuConvolutionOps")(),
        "normalization": getattr(import_module("up_platform.ops.normalization.cpu"), "CpuNormalizationOps")(),
        "activation": getattr(import_module("up_platform.ops.activation.cpu"), "CpuActivationOps")(),
        "pooling": getattr(import_module("up_platform.ops.pooling.cpu"), "CpuPoolingOps")(),
        "loss": getattr(import_module("up_platform.ops.loss.cpu"), "CpuLossOps")(),
        "attention": getattr(import_module("up_platform.ops.attention.cpu"), "CpuAttentionOps")(),
    }
    execution = getattr(import_module("up_platform." + purpose + ".cpu"), "Cpu" + purpose.capitalize())(device)
    return PlatformBackend(device, purpose, operations, execution)
