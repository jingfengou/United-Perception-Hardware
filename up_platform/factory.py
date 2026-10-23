"""Explicit selection, no hardware auto-detection or implicit provider fallback."""
from importlib import import_module
from threading import RLock

_FACTORIES = {
    "nvidia": "up_platform.backends.nvidia:build",
    "metax": "up_platform.backends.metax:build",
    "ascend": "up_platform.backends.ascend:build",
    "cambricon": "up_platform.backends.cambricon:build",
    "cpu": "up_platform.backends.cpu:build",
}
_context = None
_selection = None
_lock = RLock()


def register_backend(name, factory):
    """Register a factory(config, purpose); existing identities cannot be overwritten."""
    with _lock:
        if _context is not None:
            raise RuntimeError("Register backends before platform configuration")
        if name in _FACTORIES:
            raise ValueError(f"Backend already registered: {name}")
        _FACTORIES[name] = factory


def configure_platform(config=None, *, purpose):
    """Select once per process. Calling again with the same selection is idempotent."""
    global _context, _selection
    config = dict(config or {})
    unknown = set(config) - {"backend", "device", "requirements", "plugins", "numerics"}
    if unknown:
        raise ValueError(f"Unknown platform configuration: {sorted(unknown)}")
    if purpose not in ("training", "inference"):
        raise ValueError("purpose must be training or inference")
    if config.get("numerics", "native") != "native":
        raise ValueError("This clean implementation supports only native numerics")
    name = config.get("backend", "nvidia")
    selection = (name, purpose, config.get("device"), tuple(config.get("plugins", ())))
    with _lock:
        if _context is not None:
            selection = (name, purpose, config.get("device", _selection[2]), selection[3])
            if selection != _selection:
                raise RuntimeError("Platform is fixed after configuration; use a separate process for another purpose/vendor")
            _context.require(config.get("requirements", ()))
            return _context
        for plugin in selection[3]:
            import_module(plugin)
        factory = _FACTORIES.get(name)
        if factory is None:
            raise ValueError(f"Unknown platform: {name}; registered: {sorted(_FACTORIES)}")
        if isinstance(factory, str):
            module, attr = factory.split(":")
            factory = getattr(import_module(module), attr)
        context = factory(config, purpose)
        if context.vendor != name or context.purpose != purpose:
            raise ValueError("Backend factory returned a different vendor or purpose")
        context.require(config.get("requirements", ()))
        _context = context
        _selection = (name, purpose, str(context.device.device()), selection[3])
        return context


def get_platform():
    if _context is None:
        raise RuntimeError("Call configure_platform(..., purpose='training' or 'inference') before constructing a model")
    return _context


def is_configured():
    """Whether an explicit platform exists, without initializing one."""
    return _context is not None
