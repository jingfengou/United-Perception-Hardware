from up.accelerator import ops
"""
3D IoU Calculation and Rotated NMS
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
"""
import torch


def boxes_bev_iou_cpu(*args, **kwargs):
    return ops.boxes_bev_iou_cpu(*args, **kwargs)


def boxes_iou_bev(*args, **kwargs):
    return ops.boxes_iou_bev(*args, **kwargs)


def boxes_iou3d_gpu(*args, **kwargs):
    return ops.boxes_iou3d_gpu(*args, **kwargs)


def nms_gpu(*args, **kwargs):
    return ops.nms_gpu(*args, **kwargs)


def nms_normal_gpu(*args, **kwargs):
    return ops.nms_normal_gpu(*args, **kwargs)
