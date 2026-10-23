from contextlib import contextmanager
import importlib

import torch

from .base import TrainingBackend
from up_platform.contracts import UnsupportedOperation


class TorchTraining(TrainingBackend):
    def __init__(self, device):
        self.device = device

    def autocast(self, enabled=True, dtype=None, cache_enabled=True):
        return torch.amp.autocast(self.device.device_type, enabled=enabled, dtype=dtype, cache_enabled=cache_enabled)

    def grad_scaler(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True):
        return torch.amp.GradScaler(self.device.device_type, init_scale, growth_factor, backoff_factor, growth_interval, enabled)

    def wrap_ddp(self, module, **kwargs):
        return torch.nn.parallel.DistributedDataParallel(module, **kwargs)

    def optimizer(self, name, **kwargs):
        optimizer = getattr(torch.optim, name)
        if not isinstance(optimizer, type) or not issubclass(optimizer, torch.optim.Optimizer):
            raise ValueError("Not a torch optimizer: " + name)
        return optimizer(**kwargs)

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2.0, **kwargs):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, **kwargs)

    @contextmanager
    def validation(self, module):
        modes = tuple((child, child.training) for child in module.modules())
        try:
            module.eval()
            with torch.no_grad():
                yield module
        finally:
            for child, training in modes:
                child.training = training

    def fused_optimizer(self, name, params, **kwargs):
        if name not in ("FusedFP16SGD", "FusedFP16AdamW", "FusedZeroFP16AdamW"):
            raise UnsupportedOperation(self.device.vendor, "training", name, "Not an upstream linklink optimizer")
        try:
            optim = importlib.import_module("spring.linklink.optim")
        except ImportError as error:
            raise UnsupportedOperation(self.device.vendor, "training", name,
                                       "Matching spring.linklink SDK is required; ordinary SGD/AdamW is not a substitute") from error
        return getattr(optim, name)(params, **kwargs)
