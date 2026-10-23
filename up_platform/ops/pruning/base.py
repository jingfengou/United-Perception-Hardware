from abc import ABC, abstractmethod
from up_platform.contracts import operation


class PruningOps(ABC):
    @staticmethod
    @operation(source="msbench.scheduler.iteration", output="None.", gradients="None",
               state="Restore scheduler.iteration from the original UP checkpoint without changing masks.")
    @abstractmethod
    def set_iteration(scheduler, iteration):
        """Restore the original pruning iteration field."""
        raise NotImplementedError

    @staticmethod
    @operation(source="msbench.scheduler.get_last_prune_step", output="Original last prune step, or zero if the scheduler has no such method.",
               gradients="None")
    @abstractmethod
    def last_prune_step(scheduler):
        """Read the optional prune-step offset using the original UP zero default."""
        raise NotImplementedError


    @staticmethod
    @operation(source="pruning.module_type", output="Original MSBench class; no duplicate wrapper parameters.", gradients="Class-dependent",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def module_type(name):
        "Original MSBench class; no duplicate wrapper parameters."
        raise NotImplementedError

    @staticmethod
    @operation(source="pruning.build_scheduler", output="Original pruning scheduler and policy; same config object semantics.", gradients="Original MSBench masks/gradients",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def build_scheduler(config):
        "Original pruning scheduler and policy; same config object semantics."
        raise NotImplementedError

    @staticmethod
    @operation(source="pruning.prepare", output="Original prepared model, masks and checkpoint keys.", gradients="Original MSBench masks/gradients",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def prepare(scheduler, model):
        "Original prepared model, masks and checkpoint keys."
        raise NotImplementedError

    @staticmethod
    @operation(source="pruning.prune", output="Original in-place mask update at given iteration.", gradients="Original MSBench masks/gradients",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def prune(scheduler, model, iteration):
        "Original in-place mask update at given iteration."
        raise NotImplementedError

    @staticmethod
    @operation(source="pruning.export", output="Original in-place materialization of masks.", gradients="Original MSBench semantics",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def export(scheduler, model):
        "Original in-place materialization of masks."
        raise NotImplementedError

    @staticmethod
    @operation(source="pruning.sparse_rate", output="Original sparse-rate measurement.", gradients="None",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def sparse_rate(scheduler, model):
        "Original sparse-rate measurement."
        raise NotImplementedError

    @staticmethod
    @operation(source="pruning.set_lr_scheduler", output="Original scheduler attachment, no schedule modifications.", gradients="None",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def set_lr_scheduler(scheduler, lr_scheduler):
        "Original scheduler attachment, no schedule modifications."
        raise NotImplementedError

    @staticmethod
    @operation(source="pruning.init_optimizer", output="Original optimizer attachment and state changes.", gradients="None",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def init_optimizer(scheduler, optimizer):
        "Original optimizer attachment and state changes."
        raise NotImplementedError

