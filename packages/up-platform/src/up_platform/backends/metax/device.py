from up_platform.common.runtime import DeviceBackend
from up_platform.shared.cuda_runtime import CudaConvolutionControls


class MetaXDevice(CudaConvolutionControls, DeviceBackend):
    name = "metax"
    device_type = "cuda"
    communication_backend = "nccl"
    numerical_profiles = ("native", "portable_v1")
    extension_modules = {}
