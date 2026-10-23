"""ascend: explicit pruning bindings; registration is not validation."""
from .base import PruningOps
from up_platform.contracts import UnsupportedOperation


class AscendPruningOps(PruningOps):
    @staticmethod
    def set_iteration(scheduler, iteration):
        raise UnsupportedOperation("ascend", "common", "pruning.set_iteration", "Unimplemented vendor template")
    set_iteration.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def last_prune_step(scheduler):
        raise UnsupportedOperation("ascend", "common", "pruning.last_prune_step", "Unimplemented vendor template")
    last_prune_step.__func__.unsupported_reason = "Unimplemented vendor template"


    @staticmethod
    def module_type(name):
        raise UnsupportedOperation("ascend", "common", "pruning.module_type", "Vendor template: implementation and hardware validation required")
    module_type.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def build_scheduler(config):
        raise UnsupportedOperation("ascend", "common", "pruning.build_scheduler", "Vendor template: implementation and hardware validation required")
    build_scheduler.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def prepare(scheduler, model):
        raise UnsupportedOperation("ascend", "common", "pruning.prepare", "Vendor template: implementation and hardware validation required")
    prepare.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def prune(scheduler, model, iteration):
        raise UnsupportedOperation("ascend", "common", "pruning.prune", "Vendor template: implementation and hardware validation required")
    prune.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def export(scheduler, model):
        raise UnsupportedOperation("ascend", "common", "pruning.export", "Vendor template: implementation and hardware validation required")
    export.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def sparse_rate(scheduler, model):
        raise UnsupportedOperation("ascend", "common", "pruning.sparse_rate", "Vendor template: implementation and hardware validation required")
    sparse_rate.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def set_lr_scheduler(scheduler, lr_scheduler):
        raise UnsupportedOperation("ascend", "common", "pruning.set_lr_scheduler", "Vendor template: implementation and hardware validation required")
    set_lr_scheduler.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def init_optimizer(scheduler, optimizer):
        raise UnsupportedOperation("ascend", "common", "pruning.init_optimizer", "Vendor template: implementation and hardware validation required")
    init_optimizer.__func__.unsupported_reason = "Unimplemented vendor template"

