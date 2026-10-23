from up_platform import get_platform
import torch
import torch.nn as nn
from torch.autograd import Function
# from up.tasks.det_3d.data.data_utils import check_numpy_to_torch
import numpy as np


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def points_in_boxes_cpu(points, boxes):
    return get_platform().category('pointcloud').points_in_boxes_cpu(points, boxes)


def points_in_boxes_gpu(points, boxes):
    return get_platform().category('pointcloud').points_in_boxes_gpu(points, boxes)


class RoIAwarePool3d(nn.Module):
    def __init__(self, out_size, max_pts_each_voxel=128):
        super().__init__()
        self.out_size = out_size
        self.max_pts_each_voxel = max_pts_each_voxel

    def forward(self, rois, pts, pts_feature, pool_method='max'):
        assert pool_method in ['max', 'avg']
        return RoIAwarePool3dFunction.apply(rois, pts, pts_feature, self.out_size, self.max_pts_each_voxel, pool_method)


class RoIAwarePool3dFunction(Function):
    @staticmethod
    def apply(rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method):
        return get_platform().category('pointcloud').roiaware_pool3d(rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method)


if __name__ == '__main__':
    pass
