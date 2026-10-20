import contextlib
import importlib
from collections.abc import Mapping

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

    def __init__(self, device=None):
        self._index = None
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

    def get_rng_state(self, *args, **kwargs):
        return torch.get_rng_state() if self.api is None else self.api.get_rng_state(*args, **kwargs)

    def set_rng_state(self, state, *args, **kwargs):
        if self.api is None:
            torch.set_rng_state(state)
        else:
            self.api.set_rng_state(state, *args, **kwargs)

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
        entry = operation_entry(self, name)
        if entry["dtypes"] is not None:
            value = next((arg for arg in args if isinstance(arg, torch.Tensor) and arg.is_floating_point()), None)
            if value is not None and str(value.dtype) not in entry["dtypes"]:
                raise CapabilityError("{} operation {} does not support {}".format(
                    self.name, name, value.dtype))
        try:
            return operation(self, name)(*args, **kwargs)
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

    def capabilities(self):
        owners = (self.name,) + tuple(self.op_fallbacks)
        names = sorted({name for owner, name in OPERATIONS if owner in owners})
        return {
            "backend": self.name, "device_type": self.device_type,
            "communication_backend": self.communication_backend,
            "operations": {
                name: dict(
                    owner=next(owner for owner in owners if (owner, name) in OPERATIONS),
                    backward=next(OPERATIONS[(owner, name)]["backward"]
                                  for owner in owners if (owner, name) in OPERATIONS),
                    dtypes=operation_entry(self, name)["dtypes"],
                    variants=operation_entry(self, name)["variants"],
                    status="registered_not_hardware_validated")
                for name in names},
            "dependencies": dict(self.extension_modules),
        }
