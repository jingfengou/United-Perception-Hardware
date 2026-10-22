"""Same semantic NMS/overlap, with a common widened path for portable_v1."""
from up_platform.shared import portable


def naive_nms(dets, thresh, offset=0):
    import torch
    boxes = dets.double()
    return portable._ordered_nms(boxes, torch.arange(len(boxes), device=boxes.device),
                                 thresh, offset)


def torchvision_nms(boxes, scores, threshold):
    order = scores.argsort(descending=True, stable=True)
    return portable._ordered_nms(boxes.double(), order, threshold, 0)


def gpu_iou_overlap(b1, b2, mode="IoU"):
    return portable.gpu_iou_overlap(b1.double(), b2.double(), mode).to(b1.dtype)
