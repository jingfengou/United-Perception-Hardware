"""ascend: explicit communication bindings; registration is not validation."""
from .base import CommunicationOps
from up_platform.contracts import UnsupportedOperation


class AscendCommunicationOps(CommunicationOps):

    @staticmethod
    def is_available():
        raise UnsupportedOperation("ascend", "common", "communication.is_available", "Vendor template: implementation and hardware validation required")
    is_available.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def is_initialized():
        raise UnsupportedOperation("ascend", "common", "communication.is_initialized", "Vendor template: implementation and hardware validation required")
    is_initialized.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def get_rank(group=None):
        raise UnsupportedOperation("ascend", "common", "communication.get_rank", "Vendor template: implementation and hardware validation required")
    get_rank.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def get_world_size(group=None):
        raise UnsupportedOperation("ascend", "common", "communication.get_world_size", "Vendor template: implementation and hardware validation required")
    get_world_size.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def get_backend(group=None):
        raise UnsupportedOperation("ascend", "common", "communication.get_backend", "Vendor template: implementation and hardware validation required")
    get_backend.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def world_group():
        raise UnsupportedOperation("ascend", "common", "communication.world_group", "Vendor template: implementation and hardware validation required")
    world_group.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def sum_op():
        raise UnsupportedOperation("ascend", "common", "communication.sum_op", "Vendor template: implementation and hardware validation required")
    sum_op.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def init_process_group(backend=None, **kwargs):
        raise UnsupportedOperation("ascend", "common", "communication.init_process_group", "Vendor template: implementation and hardware validation required")
    init_process_group.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def destroy_process_group(group=None):
        raise UnsupportedOperation("ascend", "common", "communication.destroy_process_group", "Vendor template: implementation and hardware validation required")
    destroy_process_group.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def new_group(ranks=None, **kwargs):
        raise UnsupportedOperation("ascend", "common", "communication.new_group", "Vendor template: implementation and hardware validation required")
    new_group.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def barrier(group=None, async_op=False, device_ids=None):
        raise UnsupportedOperation("ascend", "common", "communication.barrier", "Vendor template: implementation and hardware validation required")
    barrier.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def all_reduce(tensor, op=None, group=None, async_op=False):
        raise UnsupportedOperation("ascend", "common", "communication.all_reduce", "Vendor template: implementation and hardware validation required")
    all_reduce.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def broadcast(tensor, src, group=None, async_op=False):
        raise UnsupportedOperation("ascend", "common", "communication.broadcast", "Vendor template: implementation and hardware validation required")
    broadcast.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def all_gather(tensor_list, tensor, group=None, async_op=False):
        raise UnsupportedOperation("ascend", "common", "communication.all_gather", "Vendor template: implementation and hardware validation required")
    all_gather.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def gather(tensor, gather_list=None, dst=0, group=None, async_op=False):
        raise UnsupportedOperation("ascend", "common", "communication.gather", "Vendor template: implementation and hardware validation required")
    gather.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def all_gather_object(object_list, obj, group=None):
        raise UnsupportedOperation("ascend", "common", "communication.all_gather_object", "Vendor template: implementation and hardware validation required")
    all_gather_object.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def broadcast_object_list(object_list, src=0, group=None, device=None):
        raise UnsupportedOperation("ascend", "common", "communication.broadcast_object_list", "Vendor template: implementation and hardware validation required")
    broadcast_object_list.__func__.unsupported_reason = "Unimplemented vendor template"

