"""Public platform contract; implementations must not import UP."""
from abc import ABC, abstractmethod
from types import MappingProxyType

from .contracts import Capability, UnsupportedOperation


class DeviceBackend(ABC):
    """Tensor/autograd storage must already be integrated with PyTorch."""
    vendor = None
    device_type = None

    @abstractmethod
    def available(self):
        """Return bool without substituting another device."""

    @abstractmethod
    def device(self, index=None):
        """Return the selected torch.device; index=None preserves current device."""

    @abstractmethod
    def set_device(self, index):
        """Bind this process to one device; no effect on other processes."""

    @abstractmethod
    def device_count(self):
        """Return visible devices, respecting the vendor visibility setting."""

    @abstractmethod
    def current_device(self):
        """Return the current device index."""

    @abstractmethod
    def properties(self, index=None):
        """Return vendor properties with name and total_memory fields."""

    @abstractmethod
    def is_accelerator(self, tensor):
        """Whether tensor belongs to this vendor's accelerator device type."""

    @abstractmethod
    def move(self, value, device=None, non_blocking=False):
        """Move Tensor/Module or nested dict/list/tuple, preserving container types."""

    @abstractmethod
    def synchronize(self, device=None):
        """Wait for work on the selected device; no extra host computation."""

    @abstractmethod
    def empty_cache(self):
        """Release unused allocator cache, not live tensor storage."""

    @abstractmethod
    def memory_allocated(self, device=None):
        """Allocated bytes on this device."""

    @abstractmethod
    def max_memory_allocated(self, device=None):
        """Peak allocated bytes since the last peak reset."""

    @abstractmethod
    def reset_peak_memory_stats(self, device=None):
        """Reset allocator peak counters only."""

    @abstractmethod
    def get_rng_state(self, device=None):
        """Return a CPU ByteTensor containing device generator state."""

    @abstractmethod
    def set_rng_state(self, state, device=None):
        """Restore device generator state without reseeding CPU/Python RNGs."""

    @abstractmethod
    def manual_seed(self, seed):
        """Seed the current device, not Python/NumPy/CPU generators."""

    @abstractmethod
    def convolution_flags(self, *, enabled=None, benchmark=None, deterministic=None, allow_tf32=None):
        """Set only explicitly supplied flags; return all current flag values."""

    @abstractmethod
    def environment(self):
        """Return runtime and SDK identity; do not infer validation from version."""


class PlatformBackend:
    """One immutable vendor/purpose selection with explicit category bindings."""
    def __init__(self, device, purpose, operations, execution):
        if purpose not in ("training", "inference"):
            raise ValueError("purpose must be training or inference")
        self._device = device
        self._purpose = purpose
        self._operations = MappingProxyType(dict(operations))
        self._execution = execution

    @property
    def vendor(self):
        return self._device.vendor

    @property
    def purpose(self):
        return self._purpose

    @property
    def device(self):
        return self._device

    @property
    def ops(self):
        return self._operations

    @property
    def execution(self):
        return self._execution

    def category(self, name):
        if name not in self._operations:
            raise UnsupportedOperation(self.vendor, self.purpose, name, "category is not registered")
        return self._operations[name]

    def capabilities(self):
        from .dependencies import available
        result = {}
        for name, implementation in self._operations.items():
            for cls in type(implementation).__mro__:
                for method, function in cls.__dict__.items():
                    if isinstance(function, staticmethod):
                        function = function.__func__
                    contract = getattr(function, "contract", None)
                    key = name + "." + method
                    if contract is not None and key not in result:
                        bound = getattr(implementation, method)
                        missing = getattr(bound, "unsupported_reason", None)
                        dependencies = tuple(getattr(bound, "dependencies", getattr(implementation, "dependencies", ())))
                        result[key] = Capability(key, f"{type(implementation).__module__}.{method}",
                                                 missing is None and self.purpose in contract.scopes,
                                                 dependencies,
                                                 dependency_available=all(available(dep) for dep in dependencies),
                                                 reason=missing or "Registered; execution and verification are separate gates.")
        return result

    def require(self, operations):
        capabilities = self.capabilities()
        for name in operations:
            capability = capabilities.get(name)
            if capability is None or not capability.registered:
                raise UnsupportedOperation(self.vendor, self.purpose, name,
                                           capability.reason if capability else "operation is not registered")
            if not capability.dependency_available:
                raise UnsupportedOperation(self.vendor, self.purpose, name,
                                           "Required dependency is not installed: " + ", ".join(capability.dependencies))
