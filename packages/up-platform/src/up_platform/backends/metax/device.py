from up_platform.common.runtime import DeviceBackend
from up_platform.shared.cuda_runtime import CudaConvolutionControls
from up_platform.errors import BackendConfigurationError


class MetaXDevice(CudaConvolutionControls, DeviceBackend):
    name = "metax"
    device_type = "cuda"
    communication_backend = "nccl"
    numerical_profiles = ("native", "portable_v1", "metax_deterministic_v1")
    extension_modules = {}

    def __init__(self, device=None):
        super().__init__(device=device)
        self._determinism = None

    def _configure_numerics(self, policy):
        if self._determinism is not None:
            if policy != "metax_deterministic_v1":
                raise BackendConfigurationError("Reset the platform to change numerical profiles")
            return
        if policy != "metax_deterministic_v1":
            return super()._configure_numerics(policy)
        if self._numerics is not None:
            raise BackendConfigurationError("Reset the platform to change numerical profiles")
        from .determinism import DeterministicConvolution
        self._determinism = DeterministicConvolution()
        self._determinism.__enter__()

    def _close_numerics(self):
        if self._determinism is not None:
            from torch.utils._python_dispatch import _get_current_dispatch_mode
            self._determinism.check_thread()
            if _get_current_dispatch_mode() is not self._determinism:
                raise BackendConfigurationError("Exit nested PyTorch dispatch modes before resetting the platform")
            self._determinism.__exit__(None, None, None)
            self._determinism = None
        super()._close_numerics()

    def numerics_state(self):
        result = super().numerics_state()
        if self._determinism is not None:
            result.update(profile="metax_deterministic_v1", accumulation="GPU FP32 per-sample convolution weight GEMM",
                          operations=dict(self._determinism.counts))
        return result
