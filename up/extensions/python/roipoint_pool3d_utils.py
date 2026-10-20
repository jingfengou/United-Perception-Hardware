from up.accelerator import ops
import torch.nn as nn
from torch.autograd import Function


class RoIPointPool3d(nn.Module):
    def __init__(self, num_sampled_points=512, pool_extra_width=1.0):
        super().__init__()
        self.num_sampled_points = num_sampled_points
        self.pool_extra_width = pool_extra_width

    def forward(self, points, point_features, boxes3d):
        """
        Args:
            points: (B, N, 3)
            point_features: (B, N, C)
            boxes3d: (B, M, 7), [x, y, z, dx, dy, dz, heading]

        Returns:
            pooled_features: (B, M, 512, 3 + C)
            pooled_empty_flag: (B, M)
        """
        return RoIPointPool3dFunction.apply(
            points, point_features, boxes3d, self.pool_extra_width, self.num_sampled_points
        )


class RoIPointPool3dFunction:
    @staticmethod
    def apply(*args, **kwargs):
        return ops.roipoint_pool3d(*args, **kwargs)



if __name__ == '__main__':
    pass
