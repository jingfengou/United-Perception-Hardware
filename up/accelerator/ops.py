"""Legacy operation names, preserving UP coordinate conventions and .apply facades."""
from up_platform.catalog import NATIVE_OPERATIONS


def call(name, *args, **kwargs):
    from . import get_accelerator
    return get_accelerator().call(name, *args, **kwargs)


def _entry(name):
    def dispatch(*args, **kwargs):
        return call(name, *args, **kwargs)
    dispatch.__name__ = name
    return dispatch


for _name in tuple(NATIVE_OPERATIONS) + ("torchvision_nms", "sync_batch_norm", "sync_batch_norm_forward"):
    globals()[_name] = _entry(_name)
