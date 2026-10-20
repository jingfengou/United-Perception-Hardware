"""Compatibility entry point; the backend owns the CUDA evaluator."""
from up.accelerator import ops


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    return ops.call("rotate_iou_eval", boxes, query_boxes, criterion, device_id)
