import contextlib
import importlib
from collections.abc import Mapping

import torch

from up_platform.errors import BackendConfigurationError, CapabilityError


class DeviceBackend:
    """Shared PyTorch implementation, with overridable hardware boundaries."""

    name = "cpu"
    device_type = "cpu"
    communication_backend = "gloo"
    op_fallbacks = ("portable",)
    extension_modules = {}
    numerical_profiles = ("native",)

    def __init__(self, device=None):
        self._index = None
        self._numerics = None
        if device is not None:
            parsed = torch.device(device)
            if parsed.type != self.device_type:
                raise BackendConfigurationError(
                    "{} backend requires {!r}, got {!r}".format(
                        self.name, self.device_type, str(device)))
            self._index = parsed.index

    @property
    def api(self):
        if self.device_type == "cpu":
            return None
        api = getattr(torch, self.device_type, None)
        if api is None:
            raise CapabilityError("{} runtime is not installed".format(self.name))
        return api

    @property
    def device(self):
        index = self._index
        if self.device_type != "cpu" and index is None:
            index = self.current_device()
        return torch.device(self.device_type, index)

    def is_available(self):
        return self.device_type == "cpu" or self.api.is_available()

    def device_count(self):
        return 1 if self.device_type == "cpu" else self.api.device_count()

    def current_device(self):
        return 0 if self.device_type == "cpu" else self.api.current_device()

    def resolve_device(self, device=None):
        if device is None:
            return self.device
        parsed = torch.device(self.device_type, device) if isinstance(device, int) else torch.device(device)
        if parsed.type != self.device_type:
            raise BackendConfigurationError("Cannot bind foreign device: " + str(device))
        return parsed

    def set_device(self, device):
        index = self.resolve_device(device).index or 0
        if self.device_type != "cpu":
            self.api.set_device(index)
            self._index = index

    def is_accelerator_tensor(self, tensor):
        return tensor.device.type == self.device_type and self.device_type != "cpu"

    def validate_environment(self, requirements):
        """Validate explicitly configured package constraints, never install them."""
        if not requirements:
            return
        import platform
        try:
            from importlib import metadata
        except ImportError:
            import importlib_metadata as metadata
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
        for package, constraint in requirements.items():
            try:
                actual = platform.python_version() if package == "python" else metadata.version(package)
            except metadata.PackageNotFoundError as exc:
                raise BackendConfigurationError("{} requires package {}".format(self.name, package)) from exc
            if Version(actual) not in SpecifierSet(constraint):
                raise BackendConfigurationError(
                    "{} requires {}{}, installed {}".format(self.name, package, constraint, actual))

    def to_device(self, value, device=None, non_blocking=False):
        target = self.device if device is None else device
        if isinstance(value, (torch.Tensor, torch.nn.Module)):
            return value.to(device=target, non_blocking=non_blocking)
        if isinstance(value, Mapping):
            return type(value)((k, self.to_device(v, target, non_blocking)) for k, v in value.items())
        if isinstance(value, tuple):
            items = [self.to_device(v, target, non_blocking) for v in value]
            return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
        if isinstance(value, list):
            return [self.to_device(v, target, non_blocking) for v in value]
        return value

    def _legacy_tensor(self, dtype, *args, **kwargs):
        device = kwargs.pop("device", self.device)
        if len(args) == 1 and not isinstance(args[0], int):
            value = args[0]
            if isinstance(value, torch.Size):
                return torch.empty(value, dtype=dtype, device=device, **kwargs)
            return torch.as_tensor(value, dtype=dtype, device=device, **kwargs)
        return torch.empty(args, dtype=dtype, device=device, **kwargs)

    def float_tensor(self, *args, **kwargs):
        return self._legacy_tensor(torch.float32, *args, **kwargs)

    def int_tensor(self, *args, **kwargs):
        return self._legacy_tensor(torch.int32, *args, **kwargs)

    def manual_seed(self, seed):
        if self.api is None:
            torch.manual_seed(seed)
        else:
            self.api.manual_seed(seed)
        if self._numerics is not None:
            self._numerics.reseed(seed)

    def get_rng_state(self, *args, **kwargs):
        value = torch.get_rng_state() if self.api is None else self.api.get_rng_state(*args, **kwargs)
        return self._numerics.state(value) if self._numerics is not None else value

    def set_rng_state(self, state, *args, **kwargs):
        restored = self._numerics.restore(state) if self._numerics is not None else None
        if restored is not None:
            seed, offset, state = restored
        if self.api is None:
            torch.set_rng_state(state)
        else:
            self.api.set_rng_state(state, *args, **kwargs)
        if restored is not None:
            self._numerics.seed, self._numerics.offset = seed, offset

    def _configure_numerics(self, policy):
        if policy not in ("native", "portable_v1"):
            raise BackendConfigurationError("Unknown numerical profile: " + str(policy))
        if self._numerics is not None:
            if policy != "portable_v1":
                raise BackendConfigurationError("Reset the accelerator to change numerical profiles")
            return
        if policy == "portable_v1":
            if policy not in self.numerical_profiles or self.device_type != "cuda":
                raise CapabilityError("{} has not enabled portable_v1 for its CUDA-shaped runtime".format(self.name))
            from up_platform.experimental.numerics.mode import PortableNumerics
            self._numerics = PortableNumerics(self.device_type, backend_name=self.name)
            self._numerics._original_mha_fastpath = torch.backends.mha.get_fastpath_enabled()
            self._numerics._original_convolution_enabled = torch.backends.cudnn.enabled
            # Convolution backward reselects its backend from the current flag.
            torch.backends.cudnn.enabled = False
            torch.backends.mha.set_fastpath_enabled(False)
            self._numerics.random_mode.__enter__()
            self._numerics.__enter__()

    def _close_numerics(self):
        if self._numerics is not None:
            from torch.overrides import _get_current_function_mode
            from torch.utils._python_dispatch import _get_current_dispatch_mode
            self._numerics._check_thread()
            if _get_current_function_mode() is not self._numerics:
                raise BackendConfigurationError("Exit nested PyTorch modes before resetting the accelerator")
            if _get_current_dispatch_mode() is not self._numerics.random_mode:
                raise BackendConfigurationError("Exit nested PyTorch dispatch modes before resetting the accelerator")
            self._numerics.__exit__(None, None, None)
            self._numerics.random_mode.__exit__(None, None, None)
            torch.backends.mha.set_fastpath_enabled(self._numerics._original_mha_fastpath)
            torch.backends.cudnn.enabled = self._numerics._original_convolution_enabled
            self._numerics = None

    def numerics_state(self):
        mode = self._numerics
        return {"profile": "portable_v1" if mode is not None else "native",
                "rng": "triton-philox4x32-10-v1" if mode is not None else "vendor-native",
                "accumulation": "selected GPU FP64 operations with FP32 boundaries" if mode is not None else "vendor-native",
                "seed": mode.seed if mode is not None else None,
                "offset": mode.offset if mode is not None else None,
                "convolution_library_enabled": torch.backends.cudnn.enabled if self.device_type == "cuda" else None,
                "operations": dict(mode.counts) if mode is not None else {}}

    def synchronize(self, *args, **kwargs):
        if self.api is not None:
            self.api.synchronize(*args, **kwargs)

    def empty_cache(self):
        if self.api is not None:
            self.api.empty_cache()

    def reset_peak_memory_stats(self):
        if self.api is not None:
            self.api.reset_peak_memory_stats()

    def memory_allocated(self, device=None):
        return 0 if self.api is None else self.api.memory_allocated(device)

    def max_memory_allocated(self, device=None):
        return 0 if self.api is None else self.api.max_memory_allocated(device)

    def get_device_properties(self, device=None):
        if self.api is None:
            raise CapabilityError("CPU has no accelerator memory properties")
        return self.api.get_device_properties(self.device if device is None else device)

    def configure_convolution(self, enabled=True):
        """Non-CUDA backends may override vendor convolution tuning."""

    def health_check(self):
        if not self.is_available():
            raise CapabilityError("{} is not available".format(self.name))
        value = torch.ones(2, device=self.device)
        if not torch.equal((value + value).cpu(), torch.full((2,), 2.)):
            raise CapabilityError("{} failed the arithmetic health check".format(self.name))
        self.synchronize()
        return True

    @contextlib.contextmanager
    def convolution_flags(self, enabled=True):
        yield

    def dependency(self, name):
        path = self.extension_modules.get(name)
        if path is None:
            raise CapabilityError("{} has no provider for {!r}".format(self.name, name))
        try:
            return importlib.import_module(path)
        except (ImportError, OSError) as exc:
            raise CapabilityError("{} requires {}: {}".format(self.name, path, exc)) from exc
