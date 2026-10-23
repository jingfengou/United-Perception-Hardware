"""Original UP detection contracts, including nonstandard outputs/gradients."""
from abc import ABC, abstractmethod
from up_platform.contracts import operation


class DetectionOps(ABC):
    @staticmethod
    @operation(source="up/extensions/python/roi_align.py:15", output="features[N,C,H,W], rois[R,5] -> [R,C,pooled_h,pooled_w]; aligned coordinates explicit.",
               gradients="features only; save features/rois/aligned", dtype="Float16/32/64 matching features/rois, contiguous.",
               empty="Empty ROI -> zero-sized output and zero feature gradient.", export="Original RoiAlign symbol/attributes. Upstream binds AVG only, not MAX.",
               state="Original autograd buffers retained; no extra numerical policy.",
               device="Original device contract; no new CPU fallback.")
    @abstractmethod
    def roi_align(features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, pooled_mode, *, aligned):
        """features[N,C,H,W], rois[R,5] -> [R,C,pooled_h,pooled_w]; aligned coordinates explicit."""
        raise NotImplementedError

    @staticmethod
    @operation(source="up/extensions/python/psroi_align.py:7", output="features[N,C,H,W], rois[R,>=5] -> [R,output_dim,group_size,group_size].",
               gradients="features only; save mapping_channel", dtype="Float16/32/64 matching features/rois, contiguous.",
               empty="Original empty behavior; upstream CPU export unavailable.", export="Original PSRoiAlign symbol.",
               state="Original autograd buffers retained; no extra numerical policy.",
               device="Original device contract; no new CPU fallback.")
    @abstractmethod
    def psroi_align(features, rois, group_size, spatial_scale, output_dim, sampling_ratio):
        """features[N,C,H,W], rois[R,>=5] -> [R,output_dim,group_size,group_size]."""
        raise NotImplementedError

    @staticmethod
    @operation(source="up/extensions/python/psroi_pool.py:12", output="features[N,C,H,W], rois[R,>=5] -> [R,output_dim,group_size,group_size].",
               gradients="features only; save mapping_channel", dtype="Float16/32/64 matching features/rois, contiguous.",
               empty="Original empty behavior; CPU placeholder only for export.", export="Original PSRoiPool symbol.",
               state="Original autograd buffers retained; no extra numerical policy.",
               device="Original device contract; no new CPU fallback.")
    @abstractmethod
    def psroi_pool(features, rois, group_size, spatial_scale, output_dim):
        """features[N,C,H,W], rois[R,>=5] -> [R,output_dim,group_size,group_size]."""
        raise NotImplementedError

    @staticmethod
    @operation(source="up/extensions/python/nms.py:9", output="dets[N,5] in score order -> device int64 indices[K]; offset explicit.",
               gradients="Not differentiable", dtype="Dets converted to float32 as upstream.",
               empty="Original native empty handling.", export="No added export support.",
               state="Original autograd buffers retained; no extra numerical policy.",
               device="Original device contract; no new CPU fallback.")
    @abstractmethod
    def naive_nms(dets, thresh, *, offset):
        """dets[N,5] in score order -> device int64 indices[K]; offset explicit."""
        raise NotImplementedError

    @staticmethod
    @operation(source="up/extensions/python/nms.py:29", output="dets[N,9] -> (filtered dets[K,9], original indices[K]); original CPU variance voting.",
               gradients="Not differentiable, detach input", dtype="Float32 dets; int64 indices.",
               empty="Original empty/filter behavior.", export="No added export support.",
               state="Original autograd buffers retained; no extra numerical policy.",
               device="Explicit original CPU operation; preserve returned device/type.")
    @abstractmethod
    def softer_nms(dets, thresh=0.0001, sigma=0.5, iou_thresh=0.3, iou_method='linear', iou_sigma=0.02, method='var_voting'):
        """dets[N,9] -> (filtered dets[K,9], original indices[K]); original CPU variance voting."""
        raise NotImplementedError

    @staticmethod
    @operation(source="up/extensions/python/iou_overlap.py:5", output="b1[N,>=4], b2[M,>=4] -> [N,M] IoU/IoF/IoS, original inclusive coordinates.",
               gradients="Not differentiable", dtype="Float16/32/64 matching dtypes, contiguous slices.",
               empty="Either input empty -> shape (0,), not (N,M).", export="No added export support.",
               state="Original autograd buffers retained; no extra numerical policy.",
               device="Original device contract; no new CPU fallback.")
    @abstractmethod
    def iou_overlap(b1, b2, mode='IoU'):
        """b1[N,>=4], b2[M,>=4] -> [N,M] IoU/IoF/IoS, original inclusive coordinates."""
        raise NotImplementedError
