"""Frozen routing context shared by explicit training and inference entry points."""
import copy
import importlib
import inspect
from types import SimpleNamespace

from .errors import CapabilityError
from .registry_factory import OPERATIONS, operation_entry, resolve


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
        owner = self.runtime._numerical_policy.provider
        if owner is None:
            return self._provider
        return SimpleNamespace(name=owner, device_type=self.device_type, purpose=self.purpose,
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
        if entry["dtypes"] is not None:
            for value in _tensor_values((args, kwargs)):
                if value.is_floating_point():
                    self.require(name, dtype=value.dtype)
        try:
            implementation = resolve(entry["implementation"])
            if entry.get("variants"):
                signature = inspect.signature(implementation)
                arguments = signature.bind(*args, **kwargs)
                arguments.apply_defaults()
                values = dict(arguments.arguments)
                for parameter in signature.parameters.values():
                    if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                        values.update(arguments.arguments.get(parameter.name, {}))
                self.require(name, variants={key: values[key] for key in entry["variants"] if key in values})
            self.runtime._numerical_policy.record_operation(name)
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
