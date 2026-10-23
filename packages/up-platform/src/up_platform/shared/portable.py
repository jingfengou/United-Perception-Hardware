"""Compatibility imports for shared category-local implementations."""
from up_platform.ops.iou.common import gpu_iou_overlap
from up_platform.ops.nms.common import naive_nms, torchvision_nms, _ordered_nms
