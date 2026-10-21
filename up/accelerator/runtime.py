import contextlib
import importlib
from collections.abc import Mapping
from types import SimpleNamespace

import torch

from .errors import BackendConfigurationError, CapabilityError
from .registry import OPERATIONS, operation, operation_entry


class TorchAccelerator:
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
            from .numerics import PortableNumerics
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

    def autocast(self, enabled=True, **kwargs):
        if enabled and self._numerics is not None:
            raise CapabilityError("portable_v1 does not support autocast; use FP32 model boundaries")
        if enabled and kwargs.get("dtype") not in (None, torch.float16, torch.bfloat16):
            raise CapabilityError("{} autocast does not support {}".format(self.name, kwargs["dtype"]))
        if not enabled and self.device_type == "cpu":
            return contextlib.ExitStack()
        if enabled and not self.is_available():
            raise CapabilityError("{} autocast requires an available device".format(self.name))
        if enabled and kwargs.get("dtype") is torch.bfloat16 and self.api is not None:
            check = getattr(self.api, "is_bf16_supported", None)
            if check is not None and not check():
                raise CapabilityError("{} device does not support bfloat16 autocast".format(self.name))
        if self.device_type not in ("cuda", "cpu") and hasattr(self.api, "amp"):
            return self.api.amp.autocast(enabled=enabled, **kwargs)
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(self.device_type, enabled=enabled, **kwargs)
        if self.api is not None and hasattr(self.api, "amp"):
            return self.api.amp.autocast(enabled=enabled, **kwargs)
        raise CapabilityError("{} does not provide autocast".format(self.name))

    def grad_scaler(self, **kwargs):
        if kwargs.get("enabled", True) and not self.is_available():
            raise CapabilityError("{} GradScaler requires an available device".format(self.name))
        if self.device_type not in ("cuda", "cpu") and hasattr(self.api, "amp"):
            return self.api.amp.GradScaler(**kwargs)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler(self.device_type, **kwargs)
        if self.api is not None and hasattr(self.api, "amp"):
            return self.api.amp.GradScaler(**kwargs)
        raise CapabilityError("{} does not provide GradScaler".format(self.name))

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

    def call(self, name, *args, **kwargs):
        provider = self._operation_backend()
        try:
            entry = operation_entry(provider, name)
        except CapabilityError as exc:
            if self._numerics is not None:
                raise CapabilityError("backend={!r}, numerics='portable_v1', operation={!r}: {}".format(
                    self.name, name, exc)) from exc
            raise
        if entry["dtypes"] is not None:
            value = next((arg for arg in args if isinstance(arg, torch.Tensor) and arg.is_floating_point()), None)
            if value is not None and str(value.dtype) not in entry["dtypes"]:
                raise CapabilityError("{} operation {} does not support {}".format(
                    self.name, name, value.dtype))
        try:
            if self._numerics is not None:
                key = "ops." + name
                self._numerics.counts[key] = self._numerics.counts.get(key, 0) + 1
            return operation(provider, name)(*args, **kwargs)
        except (ImportError, OSError) as exc:
            raise CapabilityError(
                "{} operation {!r} could not load its implementation: {}".format(
                    self.name, name, exc)) from exc

    def dependency(self, name):
        path = self.extension_modules.get(name)
        if path is None:
            raise CapabilityError("{} has no provider for {!r}".format(self.name, name))
        try:
            return importlib.import_module(path)
        except (ImportError, OSError) as exc:
            raise CapabilityError("{} requires {}: {}".format(self.name, path, exc)) from exc

    def _operation_backend(self):
        if self._numerics is None:
            return self
        return SimpleNamespace(name="portable_v1", device_type=self.device_type,
                               op_fallbacks=(self.name,) + tuple(self.op_fallbacks))

    def capabilities(self):
        provider = self._operation_backend()
        owners = (provider.name,) + tuple(provider.op_fallbacks)
        names = sorted({name for owner, name in OPERATIONS if owner in owners})
        return {
            "backend": self.name, "device_type": self.device_type,
            "communication_backend": self.communication_backend,
            "operations": {
                name: dict(
                    owner=next(owner for owner in owners if (owner, name) in OPERATIONS),
                    backward=next(OPERATIONS[(owner, name)]["backward"]
                                  for owner in owners if (owner, name) in OPERATIONS),
                    dtypes=operation_entry(provider, name)["dtypes"],
                    variants=operation_entry(provider, name)["variants"],
                    status="registered_not_hardware_validated")
                for name in names},
            "dependencies": dict(self.extension_modules),
            "numerics": self.numerics_state(),
        }
