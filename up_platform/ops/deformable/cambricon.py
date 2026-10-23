"""cambricon deformable: unimplemented contract template."""
from .base import DeformableOps
from up_platform.contracts import UnsupportedOperation


class CambriconDeformableOps(DeformableOps):
    dependencies = ()
    @staticmethod
    def deform_conv(input, offset, weight, out_channels, kernel_size, stride, pad, dilations, groups, deformable_groups):
        raise UnsupportedOperation("cambricon", "unverified-template", "deformable.deform_conv", "No native implementation validated")

    deform_conv.__func__.unsupported_reason = "Template; no native implementation"
