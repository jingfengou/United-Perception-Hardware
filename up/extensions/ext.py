"""Legacy native namespace, loaded only for the explicitly selected vendor."""
from up_platform import get_platform


def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    from up_platform.ops._native.loader import load_native
    return getattr(load_native(get_platform().vendor), name)
