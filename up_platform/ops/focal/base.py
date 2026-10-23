"""Original UP focal contracts, including nonstandard outputs/gradients."""
from abc import ABC, abstractmethod
from up_platform.contracts import operation


class FocalOps(ABC):
    @staticmethod
    @operation(source="up/extensions/python/focal_loss.py:9", output="preds[N,C], int32 targets[N], scalar normalizer -> float32[1] or [N,C] for reduction=none. Original background/ignore convention.",
               gradients="preds only; save original normalizer", dtype="Kernel float16/32/64; legacy FloatTensor output contract retained.",
               empty="Original kernel empty behavior.", export="No added export support.",
               state="Original autograd buffers retained; no extra numerical policy.",
               device="Original device contract; no new CPU fallback.")
    @abstractmethod
    def sigmoid_focal_loss(preds, targets, weight_pos, gamma, alpha, num_classes, reduction):
        """preds[N,C], int32 targets[N], scalar normalizer -> float32[1] or [N,C] for reduction=none. Original background/ignore convention."""
        raise NotImplementedError

    @staticmethod
    @operation(source="up/extensions/python/focal_loss.py:79", output="preds[N,C], int32 targets[N], normalizer -> float32[1].",
               gradients="preds only; save priors and normalizer", dtype="Kernel float16/32/64; legacy FloatTensor output contract retained.",
               empty="Original kernel empty behavior.", export="No added export support.",
               state="Original autograd buffers retained; no extra numerical policy.",
               device="Original device contract; no new CPU fallback.")
    @abstractmethod
    def softmax_focal_loss(preds, targets, weight_pos, gamma, alpha, num_classes):
        """preds[N,C], int32 targets[N], normalizer -> float32[1]."""
        raise NotImplementedError

    @staticmethod
    @operation(source="up/extensions/python/cross_focal_loss.py:9", output="preds[N,C], int32 targets[N], avoidance map -> float32[1]. Upstream early return ignores reduction in forward.",
               gradients="preds only; save avoidance map and normalizer", dtype="Kernel float16/32/64; int32 targets/map.",
               empty="Original kernel empty behavior.", export="No added export support.",
               state="Original autograd buffers retained; no extra numerical policy.",
               device="Original device contract; no new CPU fallback.")
    @abstractmethod
    def cross_sigmoid_focal_loss(preds, targets, weight_pos, gamma, alpha, num_classes, gt_class_to_avoid, reduction):
        """Return float32[1], including reduction=none, matching upstream."""
        raise NotImplementedError
