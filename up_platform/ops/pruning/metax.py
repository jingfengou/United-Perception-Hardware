"""metax: explicit pruning bindings; registration is not validation."""
from .base import PruningOps
from . import torch_impl


class MetaxPruningOps(PruningOps):
    set_iteration = staticmethod(torch_impl.set_iteration)
    last_prune_step = staticmethod(torch_impl.last_prune_step)
    dependencies = ("msbench",)
    module_type = staticmethod(torch_impl.module_type)
    build_scheduler = staticmethod(torch_impl.build_scheduler)
    prepare = staticmethod(torch_impl.prepare)
    prune = staticmethod(torch_impl.prune)
    export = staticmethod(torch_impl.export)
    sparse_rate = staticmethod(torch_impl.sparse_rate)
    set_lr_scheduler = staticmethod(torch_impl.set_lr_scheduler)
    init_optimizer = staticmethod(torch_impl.init_optimizer)

