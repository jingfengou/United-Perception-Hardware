"""Relocated from UP 52aaeb9ed27a; arithmetic and autograd state retained."""
import logging
import numpy as np
import torch
from torch.autograd import Function
from up_platform.ops._native.loader import load_native
logger = logging.getLogger(__name__)
from .geometry import check_numpy_to_torch, enlarge_box3d


class RoIPointPool3dFunction(Function):

    @staticmethod
    def forward(ctx, points, point_features, boxes3d, pool_extra_width, num_sampled_points=512, vendor=None):
        """
        Args:
            ctx:
            points: (B, N, 3)
            point_features: (B, N, C)
            boxes3d: (B, num_boxes, 7), [x, y, z, dx, dy, dz, heading]
            pool_extra_width:
            num_sampled_points:

        Returns:
            pooled_features: (B, num_boxes, 512, 3 + C)
            pooled_empty_flag: (B, num_boxes)
        """
        ctx._roipoint_pool3d_cuda = load_native(vendor).roipoint_pool3d
        assert points.shape.__len__() == 3 and points.shape[2] == 3
        batch_size, boxes_num, feature_len = (points.shape[0], boxes3d.shape[1], point_features.shape[2])
        pooled_boxes3d = enlarge_box3d(boxes3d.view(-1, 7), pool_extra_width).view(batch_size, -1, 7)
        pooled_features = point_features.new_zeros((batch_size, boxes_num, num_sampled_points, 3 + feature_len))
        pooled_empty_flag = point_features.new_zeros((batch_size, boxes_num)).int()
        ctx._roipoint_pool3d_cuda.forward(points.contiguous(), pooled_boxes3d.contiguous(), point_features.contiguous(), pooled_features, pooled_empty_flag)
        return (pooled_features, pooled_empty_flag)

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError
