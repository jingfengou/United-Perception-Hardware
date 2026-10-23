"""Explicit initialization, unlike import-time device setup."""
import copy
import importlib
from dataclasses import asdict, dataclass, field, replace

from .base.factory import resolve_backend as _resolve_backend
from .context import PlatformContext
from .errors import BackendConfigurationError, CapabilityError
from .registry_factory import BACKENDS, OPERATIONS, PURPOSES, resolve

_active = None


@dataclass
class PlatformConfig:
    backend: str = "auto"
    device: object = None
    plugins: tuple = ()
    requirements: dict = field(default_factory=dict)
    numerics: str = "native"


def configure_platform(config=None, *, purpose, reset=False):
    global _active
    if purpose not in PURPOSES:
        raise BackendConfigurationError("purpose must be training or inference")
    options = asdict(config) if isinstance(config, PlatformConfig) else dict(config or {})
    for plugin in options.get("plugins", ()):
        importlib.import_module(plugin)
    name = _resolve_backend(options.get("backend", "auto"), options.get("device"))
    if name not in BACKENDS:
        # Discover only the requested plugin; unrelated SDKs remain unloaded.
        from importlib.metadata import entry_points
        points = entry_points()
        points = points.select(group="up_platform.backends", name=name) if hasattr(points, "select") else [
            point for point in points.get("up_platform.backends", ()) if point.name == name]
        for point in points:
            point.load()()
    if name not in BACKENDS:
        raise BackendConfigurationError("Unknown accelerator {!r}; available: {}".format(name, sorted(BACKENDS)))
    if _active is not None and not reset:
        if name != _active.name or purpose != _active.purpose:
            raise BackendConfigurationError("Platform already configured as {}/{}; cannot switch to {}/{}".format(
                _active.name, _active.purpose, name, purpose))
        target = options.get("device")
        if target is not None and str(target).split(":")[0] != _active.device_type:
            raise BackendConfigurationError("Configured backend/device mismatch: " + str(target))
        _active.validate_environment(options.get("requirements", {}))
        if "numerics" in options:
            current = _active.runtime.numerics_state()["profile"]
            if options["numerics"] != current:
                raise BackendConfigurationError("Reset the platform to change numerical profiles")
        return _active
    spec = BACKENDS[name]
    factory = spec.training_factory if purpose == "training" else spec.inference_factory
    if factory is None:
        raise CapabilityError("backend={!r}, purpose={!r}: no execution backend registered".format(name, purpose))
    try:
        runtime = resolve(spec.device_factory)(device=options.get("device"))
        runtime.name = name
        # Legacy callable factories may declare their namespace only on the instance.
        providers = runtime.op_fallbacks if spec.shared_providers is None else spec.shared_providers
        runtime.op_fallbacks = tuple(providers)
        if spec.device_type is not None and runtime.device_type != spec.device_type:
            raise BackendConfigurationError("BackendSpec device_type does not match its runtime")
        spec = replace(spec, device_type=runtime.device_type, shared_providers=tuple(providers),
                       requirements=dict(spec.requirements))
        runtime.validate_environment(spec.requirements)
        runtime.validate_environment(options.get("requirements", {}))
        if purpose == "training":
            from .training.runtime.backend.factory import load_backend
        else:
            from .inference.factory import load_backend
        execution_factory = load_backend(spec)
        context = PlatformContext(runtime, spec, purpose)
        context.execution = execution_factory(context)
        context._operations = copy.deepcopy(OPERATIONS)
    except (ImportError, OSError) as exc:
        raise BackendConfigurationError("Cannot initialize {}/{}: {}".format(name, purpose, exc)) from exc
    if _active is not None:
        _active.runtime._close_numerics()
        _active = None
    runtime._configure_numerics(options.get("numerics", "native"))
    context._operations.update(runtime._numerical_policy.operations)
    _active = context
    return context


def get_platform():
    if _active is None:
        raise BackendConfigurationError("Platform is not configured; call configure_platform(..., purpose=...) first")
    return _active


def capabilities():
    return get_platform().capabilities()


def require(name, **kwargs):
    return get_platform().require(name, **kwargs)


set_ai_device = configure_platform
