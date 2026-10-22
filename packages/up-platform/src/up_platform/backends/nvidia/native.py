import importlib
from functools import lru_cache

from up_platform.errors import CapabilityError


@lru_cache(None)
def _load_binary():
    try:
        return importlib.import_module("up_platform.backends.nvidia._C")
    except (ImportError, OSError) as exc:
        raise CapabilityError(
            "NVIDIA native operations require building up-platform with UP_BUILD_BACKEND=nvidia: "
            "pip install --no-build-isolation ./packages/up-platform") from exc


def load_extension():
    from up_platform import get_platform
    platform = get_platform()
    if platform.name != "nvidia" and "nvidia" not in platform.op_fallbacks:
        raise CapabilityError("NVIDIA native extension is not enabled for " + platform.name)
    return _load_binary()


class NativeModule:
    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        return getattr(getattr(load_extension(), self.name), name)


roi_align = NativeModule("roi_align")
psroi_align = NativeModule("psroi_align")
psroi_pooling = NativeModule("psroi_pooling")
naive_nms = NativeModule("naive_nms")
softer_nms = NativeModule("softer_nms")
deform_conv_v1 = NativeModule("deform_conv_v1")
focal_loss = NativeModule("focal_loss")
cross_focal_loss = NativeModule("cross_focal_loss")
overlaps = NativeModule("overlaps")
iou3d_nms = NativeModule("iou3d_nms")
roiaware_pool3d = NativeModule("roiaware_pool3d")
roipoint_pool3d = NativeModule("roipoint_pool3d")
