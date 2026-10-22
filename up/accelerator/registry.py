"""Legacy registration signatures translated into independent platform specs."""
from up_platform.registry import (BACKENDS, BACKEND_PROBES, OPERATIONS, BackendSpec,
                                  operation, operation_entry, resolve)
from up_platform.registry import register_backend as _register_backend
from up_platform.registry import register_op as _register_op
from up_platform.catalog import TRAINING_ONLY


def register_backend(name, factory=None, replace=False, probe=None):
    def register(value):
        factory = None if isinstance(value, str) else value
        _register_backend(BackendSpec(
            name, getattr(factory, "device_type", None), value,
            "up_platform.training.backend:TrainingBackend",
            "up_platform.inference.backend:InferenceBackend",
            getattr(factory, "op_fallbacks", None)), replace=replace, probe=probe)
        return value
    return register(factory) if factory is not None else register


def register_op(backend, name, implementation=None, backward=None, replace=False,
                dtypes=None, variants=None, *, scope=None, conventions=(), dependencies=()):
    if scope is None:
        scope = "training" if name in TRAINING_ONLY else "common"
    return _register_op(backend, name, implementation, scope=scope, backward=backward,
                        replace=replace, dtypes=dtypes, variants=variants,
                        conventions=conventions, dependencies=dependencies)
