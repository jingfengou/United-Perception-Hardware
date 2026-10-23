"""Original UP pointcloud contracts, including nonstandard outputs/gradients."""
from abc import ABC, abstractmethod
from up_platform.contracts import operation


class PointcloudOps(ABC):
    @staticmethod
    @operation(source="up/tasks/det_3d/data/metrics/kitti_object_eval_python/rotate_iou.py",
               output="NumPy float32[N,K] overlap for boxes[N,5], query_boxes[K,5]; clockwise angle.",
               gradients="Not differentiable; evaluation metric.",
               device="Explicit host arrays in/out; geometry executes on selected accelerator.",
               dtype="Cast host inputs to float32 as in upstream.",
               layout="Contiguous [N,5]; center_x, center_y, width, height, clockwise angle.",
               empty="Return zeros[N,K] if N or K is zero.",
               state="Select device_id; no model state changes.",
               export="Metric only, not part of model export.")
    @abstractmethod
    def rotate_iou_eval(boxes, query_boxes, criterion=-1, device_id=0):
        """criterion=-1 IoU; 0 divide query area; 1 divide box area; otherwise intersection."""
        raise NotImplementedError

    @staticmethod
    @operation(source="up/extensions/python/roiaware_pool3d_utils.py:15", output="points[P,3], boxes[B,7] -> CPU int32[B,P]; preserve original NumPy flag.",
               gradients="Not differentiable", dtype="Explicit CPU float32 contiguous conversion.",
               empty="Original CPU loops handle empty counts.", export="No added export support.",
               state="Original autograd buffers retained; no extra numerical policy.",
               device="Explicit original CPU operation; preserve returned device/type.")
    @abstractmethod
    def points_in_boxes_cpu(points, boxes):
        """points[P,3], boxes[B,7] -> CPU int32[B,P]; preserve original NumPy flag."""
        raise NotImplementedError

    @staticmethod
    @operation(source="up/extensions/python/roiaware_pool3d_utils.py:34", output="points[B,P,3], boxes[B,T,7] -> int32[B,P], first box index or -1.",
               gradients="Not differentiable", dtype="GPU float32 contiguous inputs.",
               empty="Original native empty behavior.", export="No added export support.",
               state="Original autograd buffers retained; no extra numerical policy.",
               device="Original device contract; no new CPU fallback.")
    @abstractmethod
    def points_in_boxes_gpu(points, boxes):
        """points[B,P,3], boxes[B,T,7] -> int32[B,P], first box index or -1."""
        raise NotImplementedError

    @staticmethod
    @operation(source="up/extensions/python/roiaware_pool3d_utils.py:61", output="rois[R,7], pts[P,3], feature[P,C] -> [R,X,Y,Z,C]; max/avg.",
               gradients="pts_feature only; save argmax and voxel point indices", dtype="GPU float32 contiguous inputs; int32 state.",
               empty="Original native empty behavior.", export="No added export support.",
               state="Original autograd buffers retained; no extra numerical policy.",
               device="Original device contract; no new CPU fallback.")
    @abstractmethod
    def roiaware_pool3d(rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method):
        """rois[R,7], pts[P,3], feature[P,C] -> [R,X,Y,Z,C]; max/avg."""
        raise NotImplementedError

    @staticmethod
    @operation(source="up/extensions/python/roipoint_pool3d_utils.py:29", output="points[B,P,3], features[B,P,C], boxes[B,R,7] -> ([B,R,S,3+C], int32[B,R] flags).",
               gradients="No backward in upstream; do not fabricate gradients", dtype="GPU float32; extra_width follows original helper.",
               empty="Original native empty behavior and empty-box flags.", export="No added export support.",
               state="Original autograd buffers retained; no extra numerical policy.",
               device="Original device contract; no new CPU fallback.")
    @abstractmethod
    def roipoint_pool3d(points, point_features, boxes3d, pool_extra_width, num_sampled_points=512):
        """points[B,P,3], features[B,P,C], boxes[B,R,7] -> ([B,R,S,3+C], int32[B,R] flags)."""
        raise NotImplementedError

    @staticmethod
    @operation(source="up/extensions/python/iou3d_nms_utils.py:11", output="boxes_a[N,7], boxes_b[M,7] -> CPU[N,M]; preserve NumPy return convention.",
               gradients="Not differentiable", dtype="CPU float32, center/dimensions/heading.",
               empty="Original CPU empty behavior.", export="No added export support.",
               state="Original autograd buffers retained; no extra numerical policy.",
               device="Explicit original CPU operation; preserve returned device/type.")
    @abstractmethod
    def boxes_bev_iou_cpu(boxes_a, boxes_b):
        """boxes_a[N,7], boxes_b[M,7] -> CPU[N,M]; preserve NumPy return convention."""
        raise NotImplementedError

    @staticmethod
    @operation(source="up/extensions/python/iou3d_nms_utils.py:30", output="boxes_a[N,7], boxes_b[M,7] -> float32 GPU[N,M].",
               gradients="Not differentiable", dtype="GPU float32 contiguous.",
               empty="Original native empty behavior.", export="No added export support.",
               state="Original autograd buffers retained; no extra numerical policy.",
               device="Original device contract; no new CPU fallback.")
    @abstractmethod
    def boxes_iou_bev(boxes_a, boxes_b):
        """boxes_a[N,7], boxes_b[M,7] -> float32 GPU[N,M]."""
        raise NotImplementedError

    @staticmethod
    @operation(source="up/extensions/python/iou3d_nms_utils.py:47", output="boxes_a[N,7], boxes_b[M,7] -> GPU[N,M], original BEV plus height formula.",
               gradients="Partial autograd through Torch height/volume only; native BEV overlap has no backward", dtype="GPU float32, center/dimensions/heading.",
               empty="Original native empty behavior.", export="No added export support.",
               state="Original autograd buffers retained; no extra numerical policy.",
               device="Original device contract; no new CPU fallback.")
    @abstractmethod
    def boxes_iou3d(boxes_a, boxes_b):
        """boxes_a[N,7], boxes_b[M,7] -> GPU[N,M], original BEV plus height formula."""
        raise NotImplementedError

    @staticmethod
    @operation(source="up/extensions/python/iou3d_nms_utils.py:81", output="boxes[N,7], scores[N] -> (device int64 indices[K], None); sort and optional pre_maxsize.",
               gradients="Not differentiable", dtype="GPU float32 boxes; original score sort.",
               empty="Original native empty behavior.", export="No added export support.",
               state="Original autograd buffers retained; no extra numerical policy.",
               device="Original device contract; no new CPU fallback.")
    @abstractmethod
    def nms3d(boxes, scores, thresh, pre_maxsize=None, **kwargs):
        """boxes[N,7], scores[N] -> (device int64 indices[K], None); sort and optional pre_maxsize."""
        raise NotImplementedError

    @staticmethod
    @operation(source="up/extensions/python/iou3d_nms_utils.py:99", output="boxes[N,7], scores[N] -> (device int64 indices[K], None); original nonrotated NMS.",
               gradients="Not differentiable", dtype="GPU float32 boxes.",
               empty="Original native empty behavior.", export="No added export support.",
               state="Original autograd buffers retained; no extra numerical policy.",
               device="Original device contract; no new CPU fallback.")
    @abstractmethod
    def nms3d_normal(boxes, scores, thresh, **kwargs):
        """boxes[N,7], scores[N] -> (device int64 indices[K], None); original nonrotated NMS."""
        raise NotImplementedError
