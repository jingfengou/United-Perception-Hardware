"""Independent device foundation with explicit training/inference execution contracts."""
import copy
import importlib
import inspect
from dataclasses import asdict, dataclass, field, replace
from types import SimpleNamespace

from .errors import BackendConfigurationError, CapabilityError
from .registry import (BACKENDS, BACKEND_PROBES, OPERATIONS, PURPOSES, BackendSpec,
                       operation_entry, register_backend, register_op, resolve)
from .catalog import register_catalog

__version__ = "0.1.0"
_active = None


@dataclass
class PlatformConfig:
    backend: str = "auto"
    device: object = None
    plugins: tuple = ()
    requirements: dict = field(default_factory=dict)
    numerics: str = "native"


def _tensor_values(value):
    import torch
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_values(item)


class PlatformContext:
    def __init__(self, runtime, spec, purpose):
        self.runtime, self.spec, self._purpose = runtime, spec, purpose
        self._provider = SimpleNamespace(name=runtime.name, device_type=runtime.device_type,
                                         purpose=purpose, op_fallbacks=tuple(runtime.op_fallbacks))
        self._operations = copy.deepcopy(OPERATIONS)
        self._executions = {}
        self._dependency_checks = {}
        self.execution = None

    @property
    def purpose(self):
        return self._purpose

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name in ("grad_scaler", "wrap_model", "build_fused_optimizer", "reload_fused_optimizer"):
            if self.purpose != "training":
                raise CapabilityError("backend={!r}, purpose='inference': {} requires training".format(self.name, name))
            return getattr(self.execution, name)
        if name == "autocast":
            return self.execution.autocast
        return getattr(self.runtime, name)

    @property
    def training(self):
        if self.purpose != "training":
            raise CapabilityError("This platform was configured for inference, not training")
        return self.execution

    @property
    def inference(self):
        if self.purpose != "inference":
            raise CapabilityError("This platform was configured for training, not inference")
        return self.execution

    @property
    def ops(self):
        return self.execution.ops

    @property
    def layers(self):
        return self.execution.layers

    @property
    def sparse(self):
        return self.execution.sparse

    def _operation_backend(self):
        if self.runtime._numerics is None:
            return self._provider
        return SimpleNamespace(name="portable_v1", device_type=self.device_type, purpose=self.purpose,
                               op_fallbacks=(self._provider.name,) + self._provider.op_fallbacks)

    def _entry(self, name):
        return operation_entry(self._operation_backend(), name, purpose=self.purpose, registry=self._operations)

    def require(self, name, *, dtype=None, backward=None, variants=None, check_dependencies=False):
        entry = self._entry(name)
        prefix = "backend={!r}, purpose={!r}, operation={!r}: ".format(self.name, self.purpose, name)
        if dtype is not None and entry["dtypes"] is not None and str(dtype) not in entry["dtypes"]:
            raise CapabilityError(prefix + "does not support " + str(dtype))
        if backward is True and entry["backward"] is not True:
            raise CapabilityError(prefix + "a complete backward implementation is not declared")
        if variants:
            supported = entry.get("variants")
            for key, value in variants.items():
                if supported is not None and (key not in supported or value not in supported[key]):
                    raise CapabilityError(prefix + "unsupported variant {}={!r}".format(key, value))
        if check_dependencies:
            dependencies = entry.get("dependencies", ())
            if dependencies:
                try:
                    for module in dependencies:
                        if module.startswith("provider:"):
                            self.runtime.dependency(module.split(":", 1)[1])
                        else:
                            importlib.import_module(module)
                except (ImportError, OSError, CapabilityError) as exc:
                    self._dependency_checks[name] = False
                    raise CapabilityError(prefix + str(exc)) from exc
                self._dependency_checks[name] = True
        return entry

    def call(self, name, *args, _conventions=None, **kwargs):
        entry = self.require(name)
        if _conventions:
            for key in entry.get("conventions", ()):
                if key not in kwargs and key in _conventions:
                    kwargs[key] = _conventions[key]
        for value in _tensor_values((args, kwargs)):
            if value.is_floating_point():
                self.require(name, dtype=value.dtype)
        try:
            implementation = resolve(entry["implementation"])
            if entry.get("variants"):
                arguments = inspect.signature(implementation).bind(*args, **kwargs)
                arguments.apply_defaults()
                values = dict(arguments.arguments)
                for parameter in inspect.signature(implementation).parameters.values():
                    if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                        values.update(arguments.arguments.get(parameter.name, {}))
                self.require(name, variants={key: values[key] for key in entry["variants"] if key in values})
            if self.runtime._numerics is not None:
                key = "ops." + name
                counts = self.runtime._numerics.counts
                counts[key] = counts.get(key, 0) + 1
            result = implementation(*args, **kwargs)
        except (ImportError, OSError) as exc:
            raise CapabilityError("backend={!r}, purpose={!r}, operation={!r}: could not load implementation: {}".format(
                self.name, self.purpose, name, exc)) from exc
        except CapabilityError as exc:
            raise CapabilityError("backend={!r}, purpose={!r}, operation={!r}: {}".format(
                self.name, self.purpose, name, exc)) from exc
        self._executions[name] = self._executions.get(name, 0) + 1
        return result

    def capabilities(self):
        operations = {}
        for name in sorted({name for _, name in self._operations}):
            try:
                entry = self._entry(name)
            except CapabilityError:
                continue
            operations[name] = {key: entry[key] for key in ("owner", "scope", "backward", "dtypes", "variants")}
            operations[name].update(registered=True, dependencies_available=self._dependency_checks.get(name),
                                    executed=self._executions.get(name, 0), hardware_validated=False,
                                    status="registered_not_hardware_validated")
        return {"backend": self.name, "device_type": self.device_type, "purpose": self.purpose,
                "communication_backend": self.communication_backend, "operations": operations,
                "dependencies": dict(self.extension_modules), "numerics": self.numerics_state()}


def _resolve_backend(name, device):
    if name != "auto":
        return name
    if device is not None and str(device).split(":")[0] == "cpu":
        return "cpu"
    matches = [name for name, probe in BACKEND_PROBES.items() if probe()]
    if len(matches) > 1:
        raise BackendConfigurationError("Ambiguous accelerator probes: " + ", ".join(matches))
    if matches:
        return matches[0]
    import torch
    if hasattr(torch, "npu") or hasattr(torch, "mlu"):
        raise BackendConfigurationError("Select ascend/cambricon explicitly")
    if getattr(torch.version, "maca", None) or getattr(torch.version, "metax", None):
        raise BackendConfigurationError("Select metax explicitly; cuda is not a vendor identity")
    return "nvidia" if torch.version.cuda else "cpu"


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
            current = "portable_v1" if _active.runtime._numerics is not None else "native"
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
        execution_factory = resolve(factory)
        context = PlatformContext(runtime, spec, purpose)
        context.execution = execution_factory(context)
        context._operations = copy.deepcopy(OPERATIONS)
    except (ImportError, OSError) as exc:
        raise BackendConfigurationError("Cannot initialize {}/{}: {}".format(name, purpose, exc)) from exc
    if _active is not None:
        _active.runtime._close_numerics()
        _active = None
    runtime._configure_numerics(options.get("numerics", "native"))
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


def __getattr__(name):
    if name in ("DeviceBackend", "TorchDeviceBackend"):
        from .common.runtime import DeviceBackend
        return DeviceBackend
    raise AttributeError(name)


for _name, _device, _class in (("cpu", "cpu", "CpuDevice"), ("nvidia", "cuda", "NvidiaDevice"),
                                ("ascend", "npu", "AscendDevice"), ("cambricon", "mlu", "CambriconDevice"),
                                ("metax", "cuda", "MetaXDevice")):
    register_backend(BackendSpec(
        _name, _device, "up_platform.backends.{}.device:{}".format(_name, _class),
        "up_platform.backends.{}.training:Backend".format(_name),
        "up_platform.backends.{}.inference:Backend".format(_name)))
register_catalog()
