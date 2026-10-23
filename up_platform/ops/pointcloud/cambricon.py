"""cambricon pointcloud: unimplemented contract template."""
from .base import PointcloudOps
from up_platform.contracts import UnsupportedOperation


class CambriconPointcloudOps(PointcloudOps):
    @staticmethod
    def rotate_iou_eval(boxes, query_boxes, criterion=-1, device_id=0):
        raise UnsupportedOperation("cambricon", "common", "pointcloud.rotate_iou_eval", "GPU KITTI evaluator is not implemented or verified for this backend")

    rotate_iou_eval.__func__.unsupported_reason = "Template; no GPU evaluator validated"

    dependencies = ()
    @staticmethod
    def points_in_boxes_cpu(points, boxes):
        raise UnsupportedOperation("cambricon", "unverified-template", "pointcloud.points_in_boxes_cpu", "No native implementation validated")

    points_in_boxes_cpu.__func__.unsupported_reason = "Template; no native implementation"

    @staticmethod
    def points_in_boxes_gpu(points, boxes):
        raise UnsupportedOperation("cambricon", "unverified-template", "pointcloud.points_in_boxes_gpu", "No native implementation validated")

    points_in_boxes_gpu.__func__.unsupported_reason = "Template; no native implementation"

    @staticmethod
    def roiaware_pool3d(rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method):
        raise UnsupportedOperation("cambricon", "unverified-template", "pointcloud.roiaware_pool3d", "No native implementation validated")

    roiaware_pool3d.__func__.unsupported_reason = "Template; no native implementation"

    @staticmethod
    def roipoint_pool3d(points, point_features, boxes3d, pool_extra_width, num_sampled_points=512):
        raise UnsupportedOperation("cambricon", "unverified-template", "pointcloud.roipoint_pool3d", "No native implementation validated")

    roipoint_pool3d.__func__.unsupported_reason = "Template; no native implementation"

    @staticmethod
    def boxes_bev_iou_cpu(boxes_a, boxes_b):
        raise UnsupportedOperation("cambricon", "unverified-template", "pointcloud.boxes_bev_iou_cpu", "No native implementation validated")

    boxes_bev_iou_cpu.__func__.unsupported_reason = "Template; no native implementation"

    @staticmethod
    def boxes_iou_bev(boxes_a, boxes_b):
        raise UnsupportedOperation("cambricon", "unverified-template", "pointcloud.boxes_iou_bev", "No native implementation validated")

    boxes_iou_bev.__func__.unsupported_reason = "Template; no native implementation"

    @staticmethod
    def boxes_iou3d(boxes_a, boxes_b):
        raise UnsupportedOperation("cambricon", "unverified-template", "pointcloud.boxes_iou3d", "No native implementation validated")

    boxes_iou3d.__func__.unsupported_reason = "Template; no native implementation"

    @staticmethod
    def nms3d(boxes, scores, thresh, pre_maxsize=None, **kwargs):
        raise UnsupportedOperation("cambricon", "unverified-template", "pointcloud.nms3d", "No native implementation validated")

    nms3d.__func__.unsupported_reason = "Template; no native implementation"

    @staticmethod
    def nms3d_normal(boxes, scores, thresh, **kwargs):
        raise UnsupportedOperation("cambricon", "unverified-template", "pointcloud.nms3d_normal", "No native implementation validated")

    nms3d_normal.__func__.unsupported_reason = "Template; no native implementation"
