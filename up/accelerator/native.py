import importlib
from functools import lru_cache

from .errors import CapabilityError


@lru_cache(None)
def load_extension():
    from . import get_accelerator
    accelerator = get_accelerator()
    if accelerator.name != "nvidia" and "nvidia" not in accelerator.op_fallbacks:
        raise CapabilityError("NVIDIA native extension is not enabled for " + accelerator.name)
    for name in ("up.extensions._C", "up_extensions._C"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name not in (name, name.split(".")[0]):
                raise
    raise CapabilityError("NVIDIA native operations require building up.extensions._C with UP_BUILD_BACKEND=nvidia")


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
