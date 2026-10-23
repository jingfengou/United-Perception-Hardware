"""metax pointcloud: own compiled binary, original UP algorithm."""
from .base import PointcloudOps
from up_platform.contracts import UnsupportedOperation


class MetaxPointcloudOps(PointcloudOps):
    @staticmethod
    def rotate_iou_eval(boxes, query_boxes, criterion=-1, device_id=0):
        raise UnsupportedOperation("metax", "common", "pointcloud.rotate_iou_eval",
                                   "KITTI port has not passed original Numba exact-edge cases; see docs/platform/VALIDATION.md")

    rotate_iou_eval.__func__.unsupported_reason = "KITTI original exact-edge compatibility gate failed; GPU candidate is not enabled"

    dependencies = ("up_platform.backends._metax_ops",)
    @staticmethod
    def points_in_boxes_cpu(points, boxes):
        from ._roiaware_pool3d_utils import points_in_boxes_cpu
        return points_in_boxes_cpu(points, boxes, vendor="metax")

    @staticmethod
    def points_in_boxes_gpu(points, boxes):
        from ._roiaware_pool3d_utils import points_in_boxes_gpu
        return points_in_boxes_gpu(points, boxes, vendor="metax")

    @staticmethod
    def roiaware_pool3d(rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method):
        from ._roiaware_pool3d_utils import RoIAwarePool3dFunction
        return RoIAwarePool3dFunction.apply(rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method, "metax")

    @staticmethod
    def roipoint_pool3d(points, point_features, boxes3d, pool_extra_width, num_sampled_points=512):
        from ._roipoint_pool3d_utils import RoIPointPool3dFunction
        return RoIPointPool3dFunction.apply(points, point_features, boxes3d, pool_extra_width, num_sampled_points, "metax")

    @staticmethod
    def boxes_bev_iou_cpu(boxes_a, boxes_b):
        from ._iou3d_nms_utils import boxes_bev_iou_cpu
        return boxes_bev_iou_cpu(boxes_a, boxes_b, vendor="metax")

    @staticmethod
    def boxes_iou_bev(boxes_a, boxes_b):
        from ._iou3d_nms_utils import boxes_iou_bev
        return boxes_iou_bev(boxes_a, boxes_b, vendor="metax")

    @staticmethod
    def boxes_iou3d(boxes_a, boxes_b):
        from ._iou3d_nms_utils import boxes_iou3d_gpu
        return boxes_iou3d_gpu(boxes_a, boxes_b, vendor="metax")

    @staticmethod
    def nms3d(boxes, scores, thresh, pre_maxsize=None, **kwargs):
        from ._iou3d_nms_utils import nms_gpu
        return nms_gpu(boxes, scores, thresh, pre_maxsize, vendor="metax", **kwargs)

    @staticmethod
    def nms3d_normal(boxes, scores, thresh, **kwargs):
        from ._iou3d_nms_utils import nms_normal_gpu
        return nms_normal_gpu(boxes, scores, thresh, vendor="metax", **kwargs)
