import torch.distributed as dist


def is_available():
    return dist.is_available()


def is_initialized():
    return dist.is_initialized()


def get_rank(group=None):
    return dist.get_rank(group)


def get_world_size(group=None):
    return dist.get_world_size(group)


def get_backend(group=None):
    return dist.get_backend(group)


def world_group():
    return dist.group.WORLD


def sum_op():
    return dist.ReduceOp.SUM


def init_process_group(backend=None, **kwargs):
    return dist.init_process_group(backend=backend, **kwargs)


def destroy_process_group(group=None):
    return dist.destroy_process_group(group)


def new_group(ranks=None, **kwargs):
    return dist.new_group(ranks=ranks, **kwargs)


def barrier(group=None, async_op=False, device_ids=None):
    return dist.barrier(group=group, async_op=async_op, device_ids=device_ids)


def all_reduce(tensor, op=None, group=None, async_op=False):
    return dist.all_reduce(tensor, op=dist.ReduceOp.SUM if op is None else op, group=group, async_op=async_op)


def broadcast(tensor, src, group=None, async_op=False):
    return dist.broadcast(tensor, src, group=group, async_op=async_op)


def all_gather(tensor_list, tensor, group=None, async_op=False):
    return dist.all_gather(tensor_list, tensor, group=group, async_op=async_op)


def gather(tensor, gather_list=None, dst=0, group=None, async_op=False):
    return dist.gather(tensor, gather_list, dst, group=group, async_op=async_op)


def all_gather_object(object_list, obj, group=None):
    return dist.all_gather_object(object_list, obj, group=group)


def broadcast_object_list(object_list, src=0, group=None, device=None):
    return dist.broadcast_object_list(object_list, src, group=group, device=device)

