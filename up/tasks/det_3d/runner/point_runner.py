from up_platform import get_platform
import torch
import torch.optim
import numpy as np
from up.utils.env.gene_env import to_device
from up.runner.base_runner import BaseRunner
from up.utils.general.registry_factory import RUNNER_REGISTRY


__all__ = ['PointRunner']


@RUNNER_REGISTRY.register("point")
class PointRunner(BaseRunner):
    def __init__(self, config, work_dir='./', training=True):
        super(PointRunner, self).__init__(config, work_dir, training)

    def batch2device(self, batch):
        model_dtype = torch.float32
        if self.fp16 and self.backend == 'linklink':
            model_dtype = self.model.dtype
        for key, val in batch.items():
            if not isinstance(val, np.ndarray):
                continue
            elif key in ['frame_id', 'metadata', 'calib', 'voxel_infos', 'class_names', 'velodyne_path', 'image_meta']:
                continue
            elif key in ['image_shape']:
                batch[key] = get_platform().device.move(torch.from_numpy(val).int())
            else:
                batch[key] = get_platform().device.move(torch.from_numpy(val).float())

        if batch['points'].device != get_platform().device.device() or batch['points'].dtype != model_dtype:
            batch = to_device(batch, device=get_platform().device.device(), dtype=model_dtype)
        return batch
