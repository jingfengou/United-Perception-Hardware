import importlib

from up_platform.common.runtime import DeviceBackend


class AscendDevice(DeviceBackend):
    name = "ascend"
    device_type = "npu"
    communication_backend = "hccl"

    def __init__(self, device=None):
        importlib.import_module("torch_npu")
        super().__init__(device)
