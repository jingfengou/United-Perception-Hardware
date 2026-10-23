"""Layer factories and functional boundaries without vendor imports at definition."""
import torch


def sync_batch_norm(*args):
    from torch.nn.modules._functions import SyncBatchNorm
    return SyncBatchNorm.apply(*args)


def sync_batch_norm_forward(module, value):
    return torch.nn.SyncBatchNorm.forward(module, value)


class SyncBatchNorm(torch.nn.SyncBatchNorm):
    def forward(self, value):
        from up_platform import ops
        return ops.sync_batch_norm_forward(self, value)


_link_classes = {}


class LinkSyncBatchNorm(torch.nn.Module):
    """Resolve the native base when constructed, preserving root parameter keys."""

    def __new__(cls, *args, **kwargs):
        from up_platform import get_platform as get_accelerator
        native = get_accelerator().dependency("link").nn.SyncBatchNorm2d
        key = (cls, native)
        if key not in _link_classes:
            _link_classes[key] = type(
                cls.__name__, (cls, native),
                {"__module__": cls.__module__, "_native_link_base": native})
        return object.__new__(_link_classes[key])

    def __init__(self, *args, **kwargs):
        from up_platform import get_platform as get_accelerator
        if kwargs.get("var_mode", object()) is None:
            kwargs["var_mode"] = get_accelerator().dependency("link").syncbnVarMode_t.L2
        self._native_link_base.__init__(self, *args, **kwargs)


class _LinkSyncBNFunction:
    @staticmethod
    def apply(*args, **kwargs):
        from up_platform import ops
        return ops.call("link_sync_batch_norm", *args, **kwargs)


def _link_sync_batch_norm(*args, **kwargs):
    from up_platform import get_platform as get_accelerator
    module = get_accelerator().dependency("link")
    values = list(args)
    if len(values) > 9 and values[9] is None:
        values[9] = module.syncbnVarMode_t.L2
    return module.nn.SyncBNFunc.apply(*values, **kwargs)


class _LinkNN:
    SyncBatchNorm2d = LinkSyncBatchNorm
    SyncBNFunc = _LinkSyncBNFunction


class _VarianceMode:
    L2 = None


class _Link:
    nn = _LinkNN()
    syncbnVarMode_t = _VarianceMode()

    def __getattr__(self, name):
        def invoke(*args, **kwargs):
            from up_platform import ops
            return ops.call("legacy_link." + name, *args, **kwargs)
        return invoke

    @property
    def group(self):
        from up_platform import get_platform as get_accelerator
        return get_accelerator().dependency("link").group


link = _Link()


class OptionalModule:
    def __init__(self, provider, path=""):
        self.provider = provider
        self.path = path

    def __getattr__(self, name):
        from up_platform import get_platform as get_accelerator
        import importlib
        module = get_accelerator().dependency(self.provider)
        target = module.__name__ + ("." + self.path if self.path else "")
        return getattr(importlib.import_module(target), name)


def optional_symbol(provider, module, name):
    """Function import deferred until the optional feature is actually invoked."""
    def invoke(*args, **kwargs):
        return getattr(OptionalModule(provider, module), name)(*args, **kwargs)
    return invoke
