from up_platform.dependencies import dependency


def set_iteration(scheduler, iteration):
    scheduler.iteration = iteration


def last_prune_step(scheduler):
    return scheduler.get_last_prune_step() if hasattr(scheduler, 'get_last_prune_step') else 0


def module_type(name):
    if name != 'FrozenBatchNorm2d':
        raise ValueError('Not a declared UP MSBench type: ' + name)
    return dependency('msbench.nn.modules').FrozenBatchNorm2d


def build_scheduler(config):
    return dependency('msbench.scheduler').build_sparse_scheduler(config)


def prepare(scheduler, model):
    return scheduler.prepare_sparse_model(model=model)


def prune(scheduler, model, iteration):
    return scheduler.prune_model(model, iteration)


def export(scheduler, model):
    return scheduler.export_sparse_model(model)


def sparse_rate(scheduler, model):
    return scheduler.compute_sparse_rate(model)


def set_lr_scheduler(scheduler, lr_scheduler):
    return scheduler.set_lr_scheduler(lr_scheduler)


def init_optimizer(scheduler, optimizer):
    return scheduler.init_optimizer(optimizer)

