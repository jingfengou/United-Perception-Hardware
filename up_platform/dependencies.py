"""Load optional third-party libraries only when their capability is requested."""
from importlib import import_module, util


def dependency(name):
    try:
        return import_module(name)
    except (ImportError, OSError) as error:
        raise RuntimeError(f"Required platform dependency {name!r} is unavailable: {error}") from error


def available(name):
    try:
        return util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False

