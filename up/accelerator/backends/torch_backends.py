import contextlib
import importlib

import torch

from ..errors import BackendConfigurationError
from ..runtime import TorchAccelerator


def torchvision_nms(*args, **kwargs):
    from torchvision.ops import nms
    return nms(*args, **kwargs)


class NvidiaAccelerator(TorchAccelerator):
    name = "nvidia"
    device_type = "cuda"
    communication_backend = "nccl"
    extension_modules = {
        "sparse": "spconv.pytorch", "link": "spring.linklink",
        "quantization": "mqbench", "pruning": "msbench",
    }

    def __init__(self, device=None):
        super().__init__(device)
        if getattr(torch.version, "hip", None):
            raise BackendConfigurationError("ROCm is not the NVIDIA backend; register an explicit plugin")

    def dependency(self, name):
        if name == "sparse":
            try:
                return importlib.import_module("spconv.pytorch")
            except ModuleNotFoundError as exc:
                if exc.name not in ("spconv", "spconv.pytorch"):
                    raise
                return importlib.import_module("spconv")
        return super().dependency(name)

    def configure_convolution(self, enabled=True):
        torch.backends.cudnn.enabled = enabled

    @contextlib.contextmanager
    def convolution_flags(self, enabled=True):
        with torch.backends.cudnn.flags(enabled=enabled):
            yield


class AscendAccelerator(TorchAccelerator):
    name = "ascend"
    device_type = "npu"
    communication_backend = "hccl"

    def __init__(self, device=None):
        importlib.import_module("torch_npu")
        super().__init__(device)


class CambriconAccelerator(TorchAccelerator):
    name = "cambricon"
    device_type = "mlu"
    communication_backend = "cncl"

    def __init__(self, device=None):
        importlib.import_module("torch_mlu")
        super().__init__(device)


class MetaXAccelerator(TorchAccelerator):
    name = "metax"
    device_type = "cuda"
    communication_backend = "nccl"
    # Reusing the CUDA-shaped runtime does not opt into NVIDIA binary kernels.
    extension_modules = {}
