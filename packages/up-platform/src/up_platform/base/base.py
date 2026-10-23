import contextlib
import importlib
from collections.abc import Mapping

import torch

from up_platform.errors import BackendConfigurationError, CapabilityError
from up_platform.registry_factory import NUMERICAL_POLICIES, resolve

from up_platform.common.policy import NativePolicy


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
        self._numerical_policy = NativePolicy(self)
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
        self._numerical_policy.reseed(seed)

    def get_rng_state(self, *args, **kwargs):
        value = torch.get_rng_state() if self.api is None else self.api.get_rng_state(*args, **kwargs)
        return self._numerical_policy.wrap_rng_state(value)

    def set_rng_state(self, state, *args, **kwargs):
        def restore_native(value):
            if self.api is None:
                torch.set_rng_state(value)
            else:
                self.api.set_rng_state(value, *args, **kwargs)
        self._numerical_policy.restore_rng_state(state, restore_native)

    @property
    def _numerics(self):
        """Private legacy hook for frozen diagnostic scripts, not a routing interface."""
        return self._numerical_policy.mode

    def _configure_numerics(self, policy):
        if policy not in NUMERICAL_POLICIES:
            raise BackendConfigurationError("Unknown numerical profile: " + str(policy))
        if self._numerical_policy.profile != "native":
            if policy != self._numerical_policy.profile:
                raise BackendConfigurationError("Reset the accelerator to change numerical profiles")
            return
        selected = resolve(NUMERICAL_POLICIES[policy])(self)
        selected.start()
        self._numerical_policy = selected

    def _close_numerics(self):
        self._numerical_policy.close()
        self._numerical_policy = NativePolicy(self)

    def numerics_state(self):
        return self._numerical_policy.state()

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
