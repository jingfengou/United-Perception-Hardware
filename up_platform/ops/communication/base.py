from abc import ABC, abstractmethod
from up_platform.contracts import operation


class CommunicationOps(ABC):

    @staticmethod
    @operation(source="communication.is_available", output="bool; whether the communication library is built.", gradients="None",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def is_available():
        "bool; whether the communication library is built."
        raise NotImplementedError

    @staticmethod
    @operation(source="communication.is_initialized", output="bool; whether the default process group exists.", gradients="None",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def is_initialized():
        "bool; whether the default process group exists."
        raise NotImplementedError

    @staticmethod
    @operation(source="communication.get_rank", output="Integer rank in group; original nonmember semantics.", gradients="None",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def get_rank(group=None):
        "Integer rank in group; original nonmember semantics."
        raise NotImplementedError

    @staticmethod
    @operation(source="communication.get_world_size", output="Integer group size.", gradients="None",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def get_world_size(group=None):
        "Integer group size."
        raise NotImplementedError

    @staticmethod
    @operation(source="communication.get_backend", output="Original process group backend string.", gradients="None",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def get_backend(group=None):
        "Original process group backend string."
        raise NotImplementedError

    @staticmethod
    @operation(source="communication.world_group", output="The original WORLD process group object.", gradients="None",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def world_group():
        "The original WORLD process group object."
        raise NotImplementedError

    @staticmethod
    @operation(source="communication.sum_op", output="Original ReduceOp.SUM enum, not a replacement reduction.", gradients="None",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def sum_op():
        "Original ReduceOp.SUM enum, not a replacement reduction."
        raise NotImplementedError

    @staticmethod
    @operation(source="communication.init_process_group", output="Initialize process group with the original arguments.", gradients="None",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def init_process_group(backend=None, **kwargs):
        "Initialize process group with the original arguments."
        raise NotImplementedError

    @staticmethod
    @operation(source="communication.destroy_process_group", output="Destroy only the requested process group.", gradients="None",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def destroy_process_group(group=None):
        "Destroy only the requested process group."
        raise NotImplementedError

    @staticmethod
    @operation(source="communication.new_group", output="Original process group; identical rank ordering.", gradients="None",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def new_group(ranks=None, **kwargs):
        "Original process group; identical rank ordering."
        raise NotImplementedError

    @staticmethod
    @operation(source="communication.barrier", output="None or original asynchronous Work handle.", gradients="None",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def barrier(group=None, async_op=False, device_ids=None):
        "None or original asynchronous Work handle."
        raise NotImplementedError

    @staticmethod
    @operation(source="communication.all_reduce", output="In-place tensor reduction; None or Work.", gradients="No automatic autograd wrapper; same as torch.distributed",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def all_reduce(tensor, op=None, group=None, async_op=False):
        "In-place tensor reduction; None or Work."
        raise NotImplementedError

    @staticmethod
    @operation(source="communication.broadcast", output="In-place tensor broadcast; None or Work.", gradients="No automatic autograd wrapper",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def broadcast(tensor, src, group=None, async_op=False):
        "In-place tensor broadcast; None or Work."
        raise NotImplementedError

    @staticmethod
    @operation(source="communication.all_gather", output="Write tensor_list in rank order; None or Work.", gradients="No automatic autograd wrapper",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def all_gather(tensor_list, tensor, group=None, async_op=False):
        "Write tensor_list in rank order; None or Work."
        raise NotImplementedError

    @staticmethod
    @operation(source="communication.gather", output="Write gather_list only at destination; None or Work.", gradients="No automatic autograd wrapper",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def gather(tensor, gather_list=None, dst=0, group=None, async_op=False):
        "Write gather_list only at destination; None or Work."
        raise NotImplementedError

    @staticmethod
    @operation(source="communication.all_gather_object", output="Write rank-ordered pickled Python objects; CPU serialization is explicit.", gradients="None",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def all_gather_object(object_list, obj, group=None):
        "Write rank-ordered pickled Python objects; CPU serialization is explicit."
        raise NotImplementedError

    @staticmethod
    @operation(source="communication.broadcast_object_list", output="Write original object list with explicit serialization device.", gradients="None",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def broadcast_object_list(object_list, src=0, group=None, device=None):
        "Write original object list with explicit serialization device."
        raise NotImplementedError

