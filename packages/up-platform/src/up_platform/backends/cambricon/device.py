import importlib

from up_platform.common.runtime import DeviceBackend


class CambriconDevice(DeviceBackend):
    name = "cambricon"
    device_type = "mlu"
    communication_backend = "cncl"

    def __init__(self, device=None):
        importlib.import_module("torch_mlu")
        super().__init__(device)
