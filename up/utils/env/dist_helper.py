from up.accelerator import get_accelerator
# Standard Library
import os
import sys
import datetime
import functools

# Import from third library
import torch
from up.accelerator import distributed as dist
import torch.nn as nn
import pickle
import numpy as np
from collections import OrderedDict
from up.utils.general.global_flag import DIST_BACKEND
from up.accelerator.layers import link

MASTER_RANK = 0
_LOCAL_PROCESS_GROUP = None


def allreduce(*args, **kwargs):
    if get_world_size() <= 1:
        return
    if DIST_BACKEND.backend == 'linklink':
        return link.allreduce(*args, **kwargs)
    elif DIST_BACKEND.backend == 'dist':
        return dist.all_reduce(*args, **kwargs)
    else:
        raise NotImplementedError


def broadcast(*args, **kwargs):
    if get_world_size() <= 1:
        return
    if DIST_BACKEND.backend == 'linklink':
        return link.broadcast(*args, **kwargs)
    elif DIST_BACKEND.backend == 'dist':
        return dist.broadcast(*args, **kwargs)
    else:
        raise NotImplementedError


def allgather(*args, **kwargs):
    if get_world_size() <= 1:
        return
    if DIST_BACKEND.backend == 'linklink':
        return link.allgather(*args, **kwargs)
    elif DIST_BACKEND.backend == 'dist':
        return dist.all_gather(*args, **kwargs)
    else:
        raise NotImplementedError


def gather(*args, **kwargs):
    if get_world_size() <= 1:
        return
    if DIST_BACKEND.backend == 'linklink':
        return link.gather(*args, **kwargs)
    elif DIST_BACKEND.backend == 'dist':
        return dist.gather(*args, **kwargs)
    else:
        raise NotImplementedError


def get_dist_rank(*args, **kwargs):
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank(*args, **kwargs)


def get_link_rank(*args, **kwargs):
    if link is None:
        return 0
    if not link.is_initialized():
        return 0
    return link.get_rank(*args, **kwargs)


def get_dist_world_size(*args, **kwargs):
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(*args, **kwargs)


def get_link_world_size(*args, **kwargs):
    if link is None:
        return 1
    if not link.is_initialized():
        return 1
    return link.get_world_size(*args, **kwargs)


def get_rank_from_env():
    rank_cands = ['SLURM_PROCID', 'MV2_COMM_WORLD_RANK', 'OMPI_COMM_WORLD_RANK']
    for rank_name in rank_cands:
        if rank_name in os.environ:
            return int(os.environ[rank_name])
    return None


def get_world_size_from_env():
    ws_cands = ['SLURM_NTASKS', 'MV2_COMM_WORLD_SIZE', 'OMPI_COMM_WORLD_SIZE']
    for ws_name in ws_cands:
        if ws_name in os.environ:
            return int(os.environ[ws_name])
    return None


def dist_barrier(*args, **kwargs):
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier(*args, **kwargs)


def link_barrier(*args, **kwargs):
    if link is None:
        return
    return link.barrier(*args, **kwargs)


def barrier(*args, **kwargs):
    if get_world_size() <= 1:
        return
    if DIST_BACKEND.backend == 'linklink':
        link_barrier(*args, **kwargs)
    elif DIST_BACKEND.backend == 'dist':
        dist_barrier(*args, **kwargs)
    else:
        raise NotImplementedError


def get_rank(*args, **kwargs):
    # rank = get_rank_from_env()
    # if rank is not None:
    #     return rank
    if DIST_BACKEND.backend == 'linklink':
        return get_link_rank(*args, **kwargs)
    elif DIST_BACKEND.backend == 'dist':
        return get_dist_rank(*args, **kwargs)
    else:
        raise NotImplementedError


def get_world_size(*args, **kwargs):
    # world_size = get_world_size_from_env()
    # if world_size is not None:
    #     return world_size
    if DIST_BACKEND.backend == 'linklink':
        return get_link_world_size(*args, **kwargs)
    elif DIST_BACKEND.backend == 'dist':
        return get_dist_world_size(*args, **kwargs)
    else:
        raise NotImplementedError


def get_local_rank(*args, **kwargs):
    rank = get_rank(*args, **kwargs)
    return int(os.environ.get("LOCAL_RANK", os.environ.get(
        "SLURM_LOCALID", rank % max(1, get_accelerator().device_count()))))


def all_reduce(*args, **kwargs):
    if DIST_BACKEND.backend == 'linklink':
        return link.allreduce(*args, **kwargs)
    elif DIST_BACKEND.backend == 'dist':
        return dist.all_reduce(*args, **kwargs)
    else:
        raise NotImplementedError


class DistEnv(object):
    @property
    def rank(self):
        self._rank = get_rank()
        return self._rank

    @property
    def local_rank(self):
        self._local_rank = get_local_rank()
        return self._local_rank

    @property
    def world_size(self):
        self._world_size = get_world_size()
        return self._world_size

    def is_master(self):
        return self.rank == MASTER_RANK

    @property
    def distributed(self):
        return self.world_size > 1


env = DistEnv()


def simple_group_split(world_size, rank, num_groups):
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(link.new_group(rank_list[i]))
    group_size = world_size // num_groups
    return groups[rank // group_size]


def DistModule(model, sync=True):
    return get_accelerator().training.wrap_legacy_model(model, sync=sync, master_rank=MASTER_RANK)


def reduce_gradients(model, sync=True, allow_dead_parameter=False):
    return get_accelerator().training.reduce_gradients(model, sync=sync, allow_dead_parameter=allow_dead_parameter)


# TODO support free port
def find_free_port():
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return str(s.getsockname()[1])


def dist_init_slurm(backend=None, port='13333'):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1, pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')
    os.environ['MASTER_PORT'] = str(port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend, rank=proc_id, world_size=ntasks)
    rank = dist.get_rank()
    device = rank % get_accelerator().device_count()
    get_accelerator().set_device(device)


def dist_init_mpi(backend=None,
                  addr='localhost',
                  port="13333",
                  rank=None,
                  world_size=None):
    r"""
    Overview:
        Init the distributed training setting
    """
    os.environ['MASTER_ADDR'] = addr or os.environ.get('MASTER_ADDR', addr)
    os.environ['MASTER_PORT'] = str(port) or os.environ.get('MASTER_PORT', str(port))

    if rank is None:
        local_id = os.environ.get('OMPI_COMM_WORLD_RANK', os.environ.get('RANK'))
        if local_id is None:
            raise RuntimeError("please indicate rank explicitly in dist_init method")
        else:
            rank = int(local_id)
    if world_size is None:
        ntasks = os.environ.get('OMPI_COMM_WORLD_SIZE', os.environ.get('WORLD_SIZE'))
        if ntasks is None:
            raise RuntimeError("please indicate world_size explicitly in dist_init method")
        else:
            world_size = int(ntasks)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    rank = dist.get_rank()
    device = rank % get_accelerator().device_count()
    get_accelerator().set_device(device)


def setup_distributed_dist(port, launch):
    if launch == 'slurm':
        device = get_rank() % get_accelerator().device_count()
        get_accelerator().set_device(device)
        dist_init_slurm(port=port)
    else:
        dist_init_mpi(port=port)
        device = get_rank() % get_accelerator().device_count()
        get_accelerator().set_device(device)


def setup_distributed_link(port, launch):
    link.initialize()
    get_accelerator().set_device(link.get_local_rank())


def setup_distributed(port=33334, launch='slurm', backend='dist'):
    if backend == 'dist':
        setup_distributed_dist(port, launch)
    else:
        setup_distributed_link(port, launch)
    rank = get_rank()
    world_size = get_world_size()
    print('rank:{} world_size(gpus):{}'.format(rank, world_size))
    sys.stdout.flush()
    return rank, world_size


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    return dist.cpu_group(DIST_BACKEND.backend)


def _serialize_to_tensor(data, group):
    if DIST_BACKEND.backend == 'dist':
        device = dist.collective_device(group)
    else:
        device = torch.device("cpu")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        import logging
        logger = logging.getLogger('global')
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    if DIST_BACKEND.backend == 'dist':
        world_size = get_world_size(group=group)
    else:
        world_size = get_world_size()
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    if DIST_BACKEND.backend == 'dist':
        allgather(size_list, local_size, group=group)
    else:
        allgather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size,), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if DIST_BACKEND.backend == 'dist':
        if group is None:
            group = _get_global_gloo_group()
        if get_world_size(group) == 1:
            return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    if DIST_BACKEND.backend == 'dist':
        allgather(tensor_list, tensor, group=group)
    else:
        allgather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def gather_pk(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if DIST_BACKEND.backend == 'dist':
        if group is None:
            group = _get_global_gloo_group()
        if get_world_size(group=group) == 1:
            return [data]
        rank = get_rank(group=group)
    else:
        rank = get_rank()

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving Tensor from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [
            torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
            for _ in size_list
        ]
        if DIST_BACKEND.backend == 'dist':
            gather(tensor, tensor_list, dst=dst, group=group)
        else:
            gather(tensor_list, tensor, dst)
        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        if DIST_BACKEND.backend == 'dist':
            gather(tensor, [], dst=dst, group=group)
        else:
            gather([], tensor, dst)
        return []


def finalize():
    try:
        if DIST_BACKEND.backend == 'dist':
            if get_world_size() > 1:
                dist.destroy_process_group()
        elif DIST_BACKEND.backend == 'linklink':
            link.finalize()
        else:
            raise NotImplementedError
    except: # noqa
        pass


def pyobj2tensor(pyobj, device=None):
    """serialize picklable python object to tensor"""
    if device is None:
        device = get_accelerator().device
    storage = torch.ByteStorage.from_buffer(pickle.dumps(pyobj))
    return torch.ByteTensor(storage).to(device=device)


def tensor2pyobj(tensor):
    """deserialize tensor to picklable python object"""
    return pickle.loads(tensor.cpu().numpy().tobytes())


def all_reduce_dict(py_dict, op="sum", group=None):
    """
    Apply all reduce function for python dict object.
    NOTE: make sure that every py_dict has the same keys and values are in the same shape.

    Args:
        py_dict (dict): dict to apply all reduce op.
        op (str): operator, could be "sum" or "mean".
    """
    world_size = get_world_size()
    if world_size == 1:
        return py_dict
    # all reduce logic across different devices.
    py_key = list(py_dict.keys())
    py_key_tensor = pyobj2tensor(py_key)
    if DIST_BACKEND.backend == 'dist':
        broadcast(py_key_tensor, src=MASTER_RANK)
    else:
        broadcast(py_key_tensor, MASTER_RANK)
    py_key = tensor2pyobj(py_key_tensor)

    tensor_shapes = [py_dict[k].shape for k in py_key]
    tensor_numels = [py_dict[k].numel() for k in py_key]

    flatten_tensor = torch.cat([py_dict[k].flatten() for k in py_key])
    all_reduce(flatten_tensor)
    if op == "mean":
        flatten_tensor /= world_size
    split_tensors = [
        x.reshape(shape)
        for x, shape in zip(torch.split(flatten_tensor, tensor_numels), tensor_shapes)
    ]
    return OrderedDict({k: v for k, v in zip(py_key, split_tensors)})


def get_async_norm_states(module):
    async_norm_states = OrderedDict()
    ASYNC_NORM = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    for name, child in module.named_modules():
        if isinstance(child, ASYNC_NORM):
            for k, v in child.state_dict().items():
                async_norm_states[".".join([name, k])] = v
    return async_norm_states


def all_reduce_norm(module):
    """
    All reduce norm statistics in different devices.
    """
    if get_world_size() <= 1:
        return
    states = get_async_norm_states(module)
    states = all_reduce_dict(states, op="mean")
    module.load_state_dict(states, strict=False)


def run_shell(cmd):
    with os.popen(cmd, 'r') as fin:
        text = fin.read()
    return text


def gpu_check():
    """Backend health smoke check; no vendor-specific external stress executable."""
    result = get_accelerator().health_check()
    barrier()
    return result


def broadcast_object(obj, group=None):
    """make suare obj is picklable
    """
    if get_world_size() <= 1:
        return obj

    serialized_tensor = _serialize_to_tensor(obj, group=None).to(get_accelerator().device)
    numel = torch.IntTensor([serialized_tensor.numel()]).to(get_accelerator().device)
    broadcast(numel, MASTER_RANK)
    # serialized_tensor from storage is not resizable
    serialized_tensor = serialized_tensor.clone()
    serialized_tensor.resize_(numel)
    broadcast(serialized_tensor, MASTER_RANK)
    serialized_bytes = serialized_tensor.cpu().numpy().tobytes()
    deserialized_obj = pickle.loads(serialized_bytes)
    return deserialized_obj
