"""Opt-in portable_v1 lifecycle, routing and global PyTorch controls."""
import torch

from up_platform.common.policy import NativePolicy
from up_platform.errors import BackendConfigurationError, CapabilityError

from .mode import PortableNumerics


class PortablePolicy(NativePolicy):
    profile = "portable_v1"
    provider = "portable_v1"

    def __init__(self, runtime):
        super().__init__(runtime)
        if self.profile not in runtime.numerical_profiles or runtime.device_type != "cuda":
            raise CapabilityError("{} has not enabled portable_v1 for its CUDA-shaped runtime".format(runtime.name))
        self.mode = PortableNumerics(runtime.device_type, backend_name=runtime.name)
        self.operations = {}
        for name in ("naive_nms", "gpu_iou_overlap", "torchvision_nms"):
            self.operations[(self.provider, name)] = {
                "implementation": "up_platform.experimental.numerics.portable_numerics_ops:" + name,
                "scope": "common", "backward": False, "dtypes": ("torch.float32",), "variants": None,
                "dependencies": ("torch",), "conventions": ("offset",) if name == "naive_nms" else ()}

    def start(self):
        self._mha = torch.backends.mha.get_fastpath_enabled()
        self._convolution = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            self.mode.random_mode.__enter__()
            try:
                self.mode.__enter__()
            except BaseException:
                self.mode.random_mode.__exit__(None, None, None)
                raise
        except BaseException:
            torch.backends.mha.set_fastpath_enabled(self._mha)
            torch.backends.cudnn.enabled = self._convolution
            raise

    def close(self):
        from torch.overrides import _get_current_function_mode
        from torch.utils._python_dispatch import _get_current_dispatch_mode
        self.mode._check_thread()
        if _get_current_function_mode() is not self.mode:
            raise BackendConfigurationError("Exit nested PyTorch modes before resetting the accelerator")
        if _get_current_dispatch_mode() is not self.mode.random_mode:
            raise BackendConfigurationError("Exit nested PyTorch dispatch modes before resetting the accelerator")
        self.mode.__exit__(None, None, None)
        self.mode.random_mode.__exit__(None, None, None)
        torch.backends.mha.set_fastpath_enabled(self._mha)
        torch.backends.cudnn.enabled = self._convolution

    def reseed(self, seed):
        self.mode.reseed(seed)

    def wrap_rng_state(self, state):
        return self.mode.state(state)

    def restore_rng_state(self, state, restore_native):
        seed, offset, native = self.mode.restore(state)
        restore_native(native)
        self.mode.seed, self.mode.offset = seed, offset

    def record_operation(self, name):
        key = "ops." + name
        self.mode.counts[key] = self.mode.counts.get(key, 0) + 1

    def validate_autocast(self, enabled):
        if enabled:
            raise CapabilityError("portable_v1 does not support autocast; use FP32 model boundaries")

    def convolution_enabled(self, requested):
        return False

    def state(self):
        result = super().state()
        result.update(rng="triton-philox4x32-10-v1", seed=self.mode.seed, offset=self.mode.offset,
                      accumulation="selected GPU FP64 operations with FP32 boundaries", operations=dict(self.mode.counts))
        return result
