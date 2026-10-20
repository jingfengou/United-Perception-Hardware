"""Dependency-free registration; never import runners or vendor libraries here."""
import importlib

from .errors import CapabilityError

BACKENDS = {}
BACKEND_PROBES = {}
OPERATIONS = {}


def register_backend(name, factory=None, replace=False, probe=None):
    def register(value):
        if name in BACKENDS and not replace:
            raise ValueError("Backend already registered: " + name)
        BACKENDS[name] = value
        if probe is not None:
            BACKEND_PROBES[name] = probe
        elif replace:
            BACKEND_PROBES.pop(name, None)
        return value
    return register(factory) if factory is not None else register


def register_op(backend, name, implementation=None, backward=None, replace=False,
                dtypes=None, variants=None):
    def register(value):
        key = (backend, name)
        if key in OPERATIONS and not replace:
            raise ValueError("Operation already registered: " + repr(key))
        OPERATIONS[key] = {
            "implementation": value, "backward": backward,
            "dtypes": None if dtypes is None else tuple(str(dtype) for dtype in dtypes),
            "variants": variants,
        }
        return value
    return register(implementation) if implementation is not None else register


def resolve(value):
    if isinstance(value, str):
        module, attribute = value.split(":", 1)
        return getattr(importlib.import_module(module), attribute)
    return value


def operation_entry(backend, name):
    # Inheritance is explicit. A shared device name never implies shared kernels.
    for owner in (backend.name,) + tuple(backend.op_fallbacks):
        entry = OPERATIONS.get((owner, name))
        if entry is not None:
            return entry
    raise CapabilityError(
        "backend={!r}, device={!r}, operation={!r}: no implementation registered; "
        "register a same-device implementation in the backend plugin".format(
            backend.name, backend.device_type, name))


def operation(backend, name):
    return resolve(operation_entry(backend, name)["implementation"])
