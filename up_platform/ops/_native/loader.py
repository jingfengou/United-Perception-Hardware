"""Load only a binary built for the explicitly selected vendor."""
from functools import lru_cache
from importlib import import_module

from up_platform.contracts import UnsupportedOperation


@lru_cache(None)
def load_native(vendor):
    if vendor not in ("nvidia", "metax"):
        raise UnsupportedOperation(vendor, "common", "native_extension", "No native extension registered")
    try:
        return import_module(f"up_platform.backends._{vendor}_ops")
    except ImportError as error:
        raise UnsupportedOperation(vendor, "common", "native_extension",
                                   f"Build with UP_BUILD_BACKEND={vendor} using matching vendor PyTorch/SDK. {error}") from error
