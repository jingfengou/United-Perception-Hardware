"""United Perception. Hardware and task registration are loaded on demand."""
import importlib
import os
import sys
__version__ = "0.3.0"
_initialized = False
_initializing = False


def import_plugin():
    for path in os.environ.get("PLUGINPATH", "").split(os.pathsep):
        if not path:
            continue
        if "/" in path:
            base, module = path.rsplit('/', 1)
            sys.path.insert(0, base)
            importlib.import_module(module)
        else:
            importlib.import_module(path)


def initialize():
    """Register application components without initializing a GPU or vendor SDK."""
    global _initialized, _initializing
    if _initialized or _initializing:
        return
    _initializing = True
    try:
        import matplotlib
        matplotlib.use("Agg")
        for name in ("commands", "runner", "data", "models", "utils", "tasks", "apis"):
            module = importlib.import_module("." + name, __name__)
            globals()[name] = module
            for public in getattr(module, "__all__", [n for n in vars(module) if not n.startswith("_")]):
                globals().setdefault(public, getattr(module, public))
        import_plugin()
        _initialized = True
    finally:
        _initializing = False


def __getattr__(name):
    if name in ("accelerator", "extensions", "commands", "runner", "data", "models", "utils", "tasks", "apis"):
        module = importlib.import_module("." + name, __name__)
        globals()[name] = module
        return module
    if not _initializing:
        initialize()
        if name in globals():
            return globals()[name]
    raise AttributeError("module 'up' has no attribute {!r}".format(name))
