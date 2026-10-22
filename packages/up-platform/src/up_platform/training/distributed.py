"""DDP construction belongs to training, not to common communication."""
import torch


def wrap_model(model, device_ids=None, **kwargs):
    from up_platform import get_platform
    if get_platform().runtime.device_type == "cpu":
        device_ids = None
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids, **kwargs)
