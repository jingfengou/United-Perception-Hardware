from up.accelerator import ops
# Import from third library
import torch
from torch.autograd import Function

# Import from pod
import logging
from up.flags import ALIGNED_FLAG
logger = logging.getLogger("global")

# Import from local

# TODO use save_for_backward instead


class RoIAlignFunction:
    @staticmethod
    def apply(*args, **kwargs):
        return ops.roi_align(*args, **kwargs)

    @staticmethod
    def symbolic(g, features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, pooled_mode):
        return g.op(
            "RoiAlign",
            features,
            rois,
            spatial_scale_f=spatial_scale,
            pooled_width_i=pooled_w,
            pooled_height_i=pooled_h,
            pooled_mode_s=pooled_mode,
            sample_num_i=sampling_ratio)



class RoIAlignPool(torch.nn.Module):

    def __init__(self, pooled_h, pooled_w, sampling_ratio, pooled_mode='AVG', spatial_scale=None):
        super(RoIAlignPool, self).__init__()

        self.pooled_w = int(pooled_w)
        self.pooled_h = int(pooled_h)
        self.sampling_ratio = int(sampling_ratio)
        self.pooled_mode = pooled_mode

        if spatial_scale is not None:
            logger.warning('spatial_scale is deprecated when initializing RoIAlignPool'
                           'we move spatial_scale to forward arguments `stride` for flexiability')

    def forward(self, rois, feature, stride):
        """
        Arguments:
            rois: [N, >=5] (batch_idx, x1, y1, x2, y2)

        Notes:
            1. rois must be N*5 dim
            2. in fp16 mode, feature.dtype is fp16, but rois.dtype may not
            3. tensor must be contiguous before passing to the C code
        """
        rois = rois[:, :5].contiguous().to(dtype=feature.dtype)
        feature = feature.contiguous()
        assert rois.shape[1] == 5, rois.shape
        spatial_scale = 1.0 / stride
        return RoIAlignFunction.apply(
            feature, rois, self.pooled_h, self.pooled_w, spatial_scale, self.sampling_ratio, self.pooled_mode)

    def __repr__(self):
        s = '{name} ({pooled_h}, {pooled_w}, {sampling_ratio}, {pooled_mode})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    @classmethod
    def from_params(cls, params):
        # 7
        pooled_h = pooled_w = params['pool_size']
        # 2
        sampling_ratio = params['sampling_ratio']
        if 'pool_mode' not in params:
            pooled_mode = 'AVG'
        else:
            pooled_mode = params['pool_mode']
        return cls(pooled_h, pooled_w, sampling_ratio, pooled_mode)
