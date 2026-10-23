"""nvidia pointcloud: own compiled binary, original UP algorithm."""
from .base import PointcloudOps


class NvidiaPointcloudOps(PointcloudOps):
    @staticmethod
    def rotate_iou_eval(boxes, query_boxes, criterion=-1, device_id=0):
        from ._kitti_numba import rotate_iou_gpu_eval
        return rotate_iou_gpu_eval(boxes, query_boxes, criterion, device_id)

    rotate_iou_eval.__func__.dependencies = ('numba.cuda',)

    dependencies = ("up_platform.backends._nvidia_ops",)
    @staticmethod
    def points_in_boxes_cpu(points, boxes):
        from ._roiaware_pool3d_utils import points_in_boxes_cpu
        return points_in_boxes_cpu(points, boxes, vendor="nvidia")

    @staticmethod
    def points_in_boxes_gpu(points, boxes):
        from ._roiaware_pool3d_utils import points_in_boxes_gpu
        return points_in_boxes_gpu(points, boxes, vendor="nvidia")

    @staticmethod
    def roiaware_pool3d(rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method):
        from ._roiaware_pool3d_utils import RoIAwarePool3dFunction
        return RoIAwarePool3dFunction.apply(rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method, "nvidia")

    @staticmethod
    def roipoint_pool3d(points, point_features, boxes3d, pool_extra_width, num_sampled_points=512):
        from ._roipoint_pool3d_utils import RoIPointPool3dFunction
        return RoIPointPool3dFunction.apply(points, point_features, boxes3d, pool_extra_width, num_sampled_points, "nvidia")

    @staticmethod
    def boxes_bev_iou_cpu(boxes_a, boxes_b):
        from ._iou3d_nms_utils import boxes_bev_iou_cpu
        return boxes_bev_iou_cpu(boxes_a, boxes_b, vendor="nvidia")

    @staticmethod
    def boxes_iou_bev(boxes_a, boxes_b):
        from ._iou3d_nms_utils import boxes_iou_bev
        return boxes_iou_bev(boxes_a, boxes_b, vendor="nvidia")

    @staticmethod
    def boxes_iou3d(boxes_a, boxes_b):
        from ._iou3d_nms_utils import boxes_iou3d_gpu
        return boxes_iou3d_gpu(boxes_a, boxes_b, vendor="nvidia")

    @staticmethod
    def nms3d(boxes, scores, thresh, pre_maxsize=None, **kwargs):
        from ._iou3d_nms_utils import nms_gpu
        return nms_gpu(boxes, scores, thresh, pre_maxsize, vendor="nvidia", **kwargs)

    @staticmethod
    def nms3d_normal(boxes, scores, thresh, **kwargs):
        from ._iou3d_nms_utils import nms_normal_gpu
        return nms_normal_gpu(boxes, scores, thresh, vendor="nvidia", **kwargs)
