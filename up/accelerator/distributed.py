"""Collective and model-wrapping boundaries, independent of vendor backend names."""
import os

import torch
import torch.distributed as _dist

from .registry import register_op


def _dispatch(name, *args, **kwargs):
    from . import get_accelerator
    return get_accelerator().call("distributed." + name, *args, **kwargs)


def init_process_group(backend=None, *args, **kwargs):
    from . import get_accelerator
    accelerator = get_accelerator()
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    accelerator.set_device(local_rank)
    if backend is None:
        backend = accelerator.communication_backend
    return _dispatch("init_process_group", backend, *args, **kwargs)


def wrap_model(model, device_ids=None, **kwargs):
    return _dispatch("wrap_model", model, device_ids=device_ids, **kwargs)


def _wrap_model(model, device_ids=None, **kwargs):
    from . import get_accelerator
    if get_accelerator().device_type == "cpu":
        device_ids = None
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids, **kwargs)


def collective_device(group=None):
    return _dispatch("collective_device", group)


def _collective_device(group=None):
    from . import get_accelerator
    backend = str(_dist.get_backend(group))
    return torch.device("cpu") if backend == "gloo" else get_accelerator().device


def cpu_group(provider="dist"):
    return _dispatch("cpu_group", provider)


def _cpu_group(provider="dist"):
    if provider == "linklink":
        from .layers import link
        return link.new_group(backend="gloo") if link.get_backend() != "gloo" else link.group.WORLD
    return _dist.new_group(backend="gloo") if str(_dist.get_backend()) != "gloo" else _dist.group.WORLD


def __getattr__(name):
    return getattr(_dist, name)


def _collective(name):
    def invoke(*args, **kwargs):
        return _dispatch(name, *args, **kwargs)
    return invoke


for _name in (
        "init_process_group", "new_group", "all_reduce", "broadcast", "all_gather",
        "gather", "barrier", "reduce", "reduce_scatter", "all_gather_into_tensor",
        "broadcast_object_list", "all_gather_object", "destroy_process_group",
        "is_available", "is_initialized", "get_rank", "get_world_size", "get_backend"):
    if hasattr(_dist, _name):
        register_op("portable", "distributed." + _name, getattr(_dist, _name))
        if _name != "init_process_group":
            globals()[_name] = _collective(_name)
register_op("portable", "distributed.wrap_model", _wrap_model)
register_op("portable", "distributed.collective_device", _collective_device)
register_op("portable", "distributed.cpu_group", _cpu_group)
