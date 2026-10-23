"""Original KITTI evaluator entry point; GPU implementation is platform-owned."""
from up_platform import get_platform


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    return get_platform().category("pointcloud").rotate_iou_eval(boxes, query_boxes, criterion, device_id)
