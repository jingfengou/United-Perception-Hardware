"""nvidia deformable: own compiled binary, original UP algorithm."""
from .base import DeformableOps
from up_platform.contracts import UnsupportedOperation


class NvidiaDeformableOps(DeformableOps):
    dependencies = ("up_platform.backends._nvidia_ops",)
    @staticmethod
    def deform_conv(input, offset, weight, out_channels, kernel_size, stride, pad, dilations, groups, deformable_groups):
        import torch
        if input.device.type == "cpu" and not torch.onnx.is_in_onnx_export():
            raise UnsupportedOperation("nvidia", "common", "deformable.deform_conv", "Upstream CPU function is an export placeholder, not a numerical implementation")
        from ._deformable_conv import DeformableConvFunction
        return DeformableConvFunction.apply(input, offset, weight, out_channels, kernel_size, stride, pad, dilations, groups, deformable_groups, "nvidia")
