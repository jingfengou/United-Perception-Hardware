"""metax detection: own compiled binary, original UP algorithm."""
from .base import DetectionOps
from up_platform.contracts import UnsupportedOperation


class MetaxDetectionOps(DetectionOps):
    dependencies = ("up_platform.backends._metax_ops",)
    @staticmethod
    def roi_align(features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, pooled_mode, *, aligned):
        import torch
        if features.device.type == "cpu" and not torch.onnx.is_in_onnx_export():
            raise UnsupportedOperation("metax", "common", "detection.roi_align", "Upstream CPU function is an export placeholder, not a numerical implementation")
        if pooled_mode != "AVG":
            raise UnsupportedOperation("metax", "common", "detection.roi_align", "Upstream binds AVG only; MAX has no compiled symbol")
        from ._roi_align import RoIAlignFunction
        return RoIAlignFunction.apply(features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, pooled_mode, aligned, "metax")

    @staticmethod
    def psroi_align(features, rois, group_size, spatial_scale, output_dim, sampling_ratio):
        from ._psroi_align import PSRoIAlignFunction
        return PSRoIAlignFunction.apply(features, rois, group_size, spatial_scale, output_dim, sampling_ratio, "metax")

    @staticmethod
    def psroi_pool(features, rois, group_size, spatial_scale, output_dim):
        import torch
        if features.device.type == "cpu" and not torch.onnx.is_in_onnx_export():
            raise UnsupportedOperation("metax", "common", "detection.psroi_pool", "Upstream CPU function is an export placeholder, not a numerical implementation")
        from ._psroi_pool import PSRoIPoolFunction
        return PSRoIPoolFunction.apply(features, rois, group_size, spatial_scale, output_dim, "metax")

    @staticmethod
    def naive_nms(dets, thresh, *, offset):
        from ._nms import naive_nms
        return naive_nms(dets, thresh, offset=offset, vendor="metax")

    @staticmethod
    def softer_nms(dets, thresh=0.0001, sigma=0.5, iou_thresh=0.3, iou_method='linear', iou_sigma=0.02, method='var_voting'):
        from ._nms import softer_nms
        return softer_nms(dets, thresh, sigma, iou_thresh, iou_method, iou_sigma, method, vendor="metax")

    @staticmethod
    def iou_overlap(b1, b2, mode='IoU'):
        from ._iou_overlap import gpu_iou_overlap
        return gpu_iou_overlap(b1, b2, mode, vendor="metax")
