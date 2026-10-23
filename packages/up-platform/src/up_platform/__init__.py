"""Hardware-independent platform API with explicit device and purpose selection."""
from .errors import BackendConfigurationError, CapabilityError
from .registry_factory import (BACKENDS, BACKEND_PROBES, OPERATIONS, PURPOSES, BackendSpec,
                               INFERENCE_FACTORY, TRAINING_FACTORY,
                               operation_entry, register_backend, register_op, resolve)
from . import registry  # Legacy public module address, sharing the same registry.
from .base import metadata as _backend_metadata
from .context import PlatformContext
from .set_ai_device import PlatformConfig, configure_platform, get_platform, capabilities, require
from .catalog import register_catalog

__version__ = "0.1.0"


def __getattr__(name):
    if name in ("DeviceBackend", "TorchDeviceBackend"):
        from .base.base import DeviceBackend
        return DeviceBackend
    raise AttributeError(name)


register_catalog()
