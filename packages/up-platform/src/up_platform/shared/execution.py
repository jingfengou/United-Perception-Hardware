"""Shared routing surface, without importing either purpose's implementation."""
from up_platform.common.precision import autocast


class OperationNamespace:
    def __init__(self, context):
        self.context = context

    def call(self, name, *args, **kwargs):
        return self.context.call(name, *args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        def invoke(*args, **kwargs):
            return self.context.call(name, *args, **kwargs)
        return invoke


class OperationBackend:
    def __init__(self, context):
        self.context = context
        self.runtime = context.runtime
        self._ops = OperationNamespace(context)

    def call(self, name, *args, **kwargs):
        return self.context.call(name, *args, **kwargs)

    def autocast(self, enabled=True, **kwargs):
        override = getattr(self.runtime, "autocast", None)
        if override is not None and not getattr(override, "_platform_compat", False):
            return override(enabled=enabled, **kwargs)
        return autocast(self.runtime, enabled=enabled, **kwargs)

    @property
    def ops(self):
        return self._ops

    @property
    def layers(self):
        from up_platform.shared import layers
        return layers

    @property
    def sparse(self):
        from up_platform.shared import sparse
        return sparse
