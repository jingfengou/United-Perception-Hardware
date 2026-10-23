"""cambricon detection: unimplemented contract template."""
from .base import DetectionOps
from up_platform.contracts import UnsupportedOperation


class CambriconDetectionOps(DetectionOps):
    dependencies = ()
    @staticmethod
    def roi_align(features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, pooled_mode, *, aligned):
        raise UnsupportedOperation("cambricon", "unverified-template", "detection.roi_align", "No native implementation validated")

    roi_align.__func__.unsupported_reason = "Template; no native implementation"

    @staticmethod
    def psroi_align(features, rois, group_size, spatial_scale, output_dim, sampling_ratio):
        raise UnsupportedOperation("cambricon", "unverified-template", "detection.psroi_align", "No native implementation validated")

    psroi_align.__func__.unsupported_reason = "Template; no native implementation"

    @staticmethod
    def psroi_pool(features, rois, group_size, spatial_scale, output_dim):
        raise UnsupportedOperation("cambricon", "unverified-template", "detection.psroi_pool", "No native implementation validated")

    psroi_pool.__func__.unsupported_reason = "Template; no native implementation"

    @staticmethod
    def naive_nms(dets, thresh, *, offset):
        raise UnsupportedOperation("cambricon", "unverified-template", "detection.naive_nms", "No native implementation validated")

    naive_nms.__func__.unsupported_reason = "Template; no native implementation"

    @staticmethod
    def softer_nms(dets, thresh=0.0001, sigma=0.5, iou_thresh=0.3, iou_method='linear', iou_sigma=0.02, method='var_voting'):
        raise UnsupportedOperation("cambricon", "unverified-template", "detection.softer_nms", "No native implementation validated")

    softer_nms.__func__.unsupported_reason = "Template; no native implementation"

    @staticmethod
    def iou_overlap(b1, b2, mode='IoU'):
        raise UnsupportedOperation("cambricon", "unverified-template", "detection.iou_overlap", "No native implementation validated")

    iou_overlap.__func__.unsupported_reason = "Template; no native implementation"
