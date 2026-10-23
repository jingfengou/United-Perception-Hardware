"""metax: explicit communication bindings; registration is not validation."""
from .base import CommunicationOps
from . import torch_impl


class MetaxCommunicationOps(CommunicationOps):
    dependencies = ("torch.distributed",)
    is_available = staticmethod(torch_impl.is_available)
    is_initialized = staticmethod(torch_impl.is_initialized)
    get_rank = staticmethod(torch_impl.get_rank)
    get_world_size = staticmethod(torch_impl.get_world_size)
    get_backend = staticmethod(torch_impl.get_backend)
    world_group = staticmethod(torch_impl.world_group)
    sum_op = staticmethod(torch_impl.sum_op)
    init_process_group = staticmethod(torch_impl.init_process_group)
    destroy_process_group = staticmethod(torch_impl.destroy_process_group)
    new_group = staticmethod(torch_impl.new_group)
    barrier = staticmethod(torch_impl.barrier)
    all_reduce = staticmethod(torch_impl.all_reduce)
    broadcast = staticmethod(torch_impl.broadcast)
    all_gather = staticmethod(torch_impl.all_gather)
    gather = staticmethod(torch_impl.gather)
    all_gather_object = staticmethod(torch_impl.all_gather_object)
    broadcast_object_list = staticmethod(torch_impl.broadcast_object_list)

