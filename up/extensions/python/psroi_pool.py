from up_platform import get_platform
# Import from pod
# Import from third library
import torch
from torch.autograd import Function

from up.utils.general.log_helper import default_logger as logger

# Import from local


class PSRoIPoolFunction(Function):
    @staticmethod
    def apply(features, rois, group_size, spatial_scale, output_dim):
        return get_platform().category('detection').psroi_pool(features, rois, group_size, spatial_scale, output_dim)


class PSRoIPool(torch.nn.Module):

    def __init__(self, group_size, output_dim=None, spatial_scale=None):
        super(PSRoIPool, self).__init__()

        self.group_size = int(group_size)
        if spatial_scale is not None:
            logger.warning('`spatial_scale` is deprecated in PSRoIPool.__ini__, '
                           'we move `spatial_scale` to `forward` arguments `stride` for flexiability')

        if output_dim is not None:
            logger.warning('`output_dim` is deprecated in PSRoIPool.__ini__, '
                           'we will calculate `output_dim` by chanels of pooled '
                           '`features` and `group_size` dynamically')

    def forward(self, rois, features, stride):
        """
        Arguments:
            rois: [N, >=5] (batch_idx, x1, y1, x2, y2)

        Notes:
            1. rois must be N*5 dim
            2. in fp16 mode, feature.dtype is fp16, but rois.dtype may not
            3. tensor must be contiguous before passing to the C code
        """
        rois = rois[:, :5].contiguous().to(dtype=features.dtype)
        features = features.contiguous()
        assert rois.shape[1] == 5, rois.shape
        spatial_scale = 1.0 / stride
        output_dim = features.shape[1] // self.group_size**2
        # In ONNX context, tensor.shape is type of tensor, while symbolic of PSRoIPool requires
        # the argumement of output_dim is int
        if torch.is_tensor(output_dim):
            output_dim = output_dim.item()
        assert self.group_size**2 * output_dim == features.shape[1]
        return PSRoIPoolFunction.apply(features, rois, self.group_size, spatial_scale, output_dim)

    def __repr__(self):
        s = '{name} ({group_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    @classmethod
    def from_params(cls, params):
        group_size = params['pool_size']
        return cls(group_size)
