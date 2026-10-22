"""Legacy plugin base; runtime implementation lives in up_platform.common."""
from up_platform.common.runtime import DeviceBackend


class TorchAccelerator(DeviceBackend):
    def autocast(self, enabled=True, **kwargs):
        from up_platform.common.precision import autocast
        return autocast(self, enabled=enabled, **kwargs)

    def grad_scaler(self, **kwargs):
        return self._legacy_context().training.grad_scaler(**kwargs)

    def call(self, name, *args, **kwargs):
        from . import _CompatibilityContext
        return _CompatibilityContext(self._legacy_context()).call(name, *args, **kwargs)

    def capabilities(self):
        return self._legacy_context().capabilities()

    def _legacy_context(self):
        import up_platform
        from up_platform.training.backend import TrainingBackend
        try:
            context = up_platform.get_platform()
            if context.runtime is self:
                return context
        except up_platform.BackendConfigurationError:
            pass
        spec = up_platform.BackendSpec(self.name, self.device_type, type(self), shared_providers=self.op_fallbacks)
        context = up_platform.PlatformContext(self, spec, "training")
        context.execution = TrainingBackend(context)
        return context

    autocast._platform_compat = True
    grad_scaler._platform_compat = True
