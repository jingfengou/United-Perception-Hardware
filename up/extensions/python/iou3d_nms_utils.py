from up_platform import get_platform
"""
3D IoU Calculation and Rotated NMS
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
"""
import torch


def boxes_bev_iou_cpu(boxes_a, boxes_b):
    return get_platform().category('pointcloud').boxes_bev_iou_cpu(boxes_a, boxes_b)


def boxes_iou_bev(boxes_a, boxes_b):
    return get_platform().category('pointcloud').boxes_iou_bev(boxes_a, boxes_b)


def boxes_iou3d_gpu(boxes_a, boxes_b):
    return get_platform().category('pointcloud').boxes_iou3d(boxes_a, boxes_b)


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    return get_platform().category('pointcloud').nms3d(boxes, scores, thresh, pre_maxsize, **kwargs)


def nms_normal_gpu(boxes, scores, thresh, **kwargs):
    return get_platform().category('pointcloud').nms3d_normal(boxes, scores, thresh, **kwargs)
