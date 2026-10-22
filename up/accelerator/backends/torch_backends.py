"""Legacy device class names; implementations are shared with up_platform."""
from up_platform.backends.nvidia.device import NvidiaDevice
from up_platform.backends.ascend.device import AscendDevice
from up_platform.backends.cambricon.device import CambriconDevice
from up_platform.backends.metax.device import MetaXDevice
from up_platform.backends.nvidia.ops.vision import torchvision_nms
from up.accelerator.runtime import TorchAccelerator


class NvidiaAccelerator(NvidiaDevice, TorchAccelerator):
    pass


class AscendAccelerator(AscendDevice, TorchAccelerator):
    pass


class CambriconAccelerator(CambriconDevice, TorchAccelerator):
    pass


class MetaXAccelerator(MetaXDevice, TorchAccelerator):
    pass
