import importlib

import torch

from up_platform.base.base import DeviceBackend
from up_platform.errors import BackendConfigurationError
from up_platform.base.cuda import CudaConvolutionControls


class NvidiaDevice(CudaConvolutionControls, DeviceBackend):
    name = "nvidia"
    device_type = "cuda"
    communication_backend = "nccl"
    numerical_profiles = ("native", "portable_v1")
    extension_modules = {"sparse": "spconv.pytorch", "link": "spring.linklink",
                         "quantization": "mqbench", "pruning": "msbench"}

    def __init__(self, device=None):
        super().__init__(device)
        if getattr(torch.version, "hip", None):
            raise BackendConfigurationError("ROCm is not the NVIDIA backend; register an explicit plugin")
        if (getattr(torch.version, "maca", None) or getattr(torch.version, "metax", None)
                or any(name in torch.__version__.lower() for name in ("metax", "maca"))):
            raise BackendConfigurationError("MetaX PyTorch must select the metax backend, not nvidia")

    def dependency(self, name):
        if name == "sparse":
            try:
                return importlib.import_module("spconv.pytorch")
            except ModuleNotFoundError as exc:
                if exc.name not in ("spconv", "spconv.pytorch"):
                    raise
                return importlib.import_module("spconv")
        return super().dependency(name)
