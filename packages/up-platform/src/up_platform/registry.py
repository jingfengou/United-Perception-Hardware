"""Metadata-only registration. Importing this module never initializes hardware."""
import importlib
from dataclasses import dataclass, field

from .errors import BackendConfigurationError, CapabilityError

PURPOSES = ("training", "inference")
SCOPES = ("common",) + PURPOSES
BACKENDS = {}
BACKEND_PROBES = {}
OPERATIONS = {}


@dataclass(frozen=True)
class BackendSpec:
    name: str
    device_type: str
    device_factory: object
    training_factory: object = None
    inference_factory: object = None
    shared_providers: tuple = ("portable",)
    requirements: dict = field(default_factory=dict)


def register_backend(spec, *, replace=False, probe=None):
    if not isinstance(spec, BackendSpec):
        raise TypeError("register_backend requires a BackendSpec")
    if spec.name in BACKENDS and not replace:
        raise ValueError("Backend already registered: " + spec.name)
    BACKENDS[spec.name] = spec
    if probe is not None:
        BACKEND_PROBES[spec.name] = probe
    elif replace:
        BACKEND_PROBES.pop(spec.name, None)
    return spec


def register_op(backend, name, implementation=None, *, scope, backward=None,
                replace=False, dtypes=None, variants=None, conventions=(), dependencies=()):
    if scope not in SCOPES:
        raise ValueError("Operation scope must be common, training or inference")

    def register(value):
        key = (backend, name)
        previous = OPERATIONS.get(key)
        implementations = dict(previous.get("implementations", {})) if previous else {}
        if previous and not implementations:
            implementations[previous.get("scope", "common")] = previous
        if scope in implementations and not replace:
            raise ValueError("Operation already registered: " + repr((backend, name, scope)))
        entry = {"implementation": value, "backward": backward, "scope": scope,
                 "dtypes": None if dtypes is None else tuple(str(dtype) for dtype in dtypes),
                 "variants": variants, "conventions": tuple(conventions),
                 "dependencies": tuple(dependencies)}
        implementations[scope] = entry
        representative = implementations.get("common", implementations.get("training", entry))
        OPERATIONS[key] = dict(representative, implementations=implementations)
        return value
    return register(implementation) if implementation is not None else register


def resolve(value):
    if isinstance(value, str):
        module, attribute = value.split(":", 1)
        value = importlib.import_module(module)
        for part in attribute.split("."):
            value = getattr(value, part)
    return value


def operation_entry(backend, name, *, purpose=None, registry=None):
    purpose = purpose or getattr(backend, "purpose", "training")
    if purpose not in PURPOSES:
        raise BackendConfigurationError("Invalid platform purpose: " + str(purpose))
    registry = OPERATIONS if registry is None else registry
    for owner in (backend.name,) + tuple(backend.op_fallbacks):
        record = registry.get((owner, name))
        if record is None:
            continue
        implementations = record.get("implementations") or {record.get("scope", "common"): record}
        entry = implementations.get(purpose, implementations.get("common"))
        if entry is not None:
            return dict(entry, owner=owner)
    raise CapabilityError(
        "backend={!r}, device={!r}, purpose={!r}, operation={!r}: no implementation registered; "
        "register a same-device implementation for this purpose".format(
            backend.name, backend.device_type, purpose, name))


def operation(backend, name, *, purpose=None, registry=None):
    return resolve(operation_entry(backend, name, purpose=purpose, registry=registry)["implementation"])
