import torch

from up_platform.errors import CapabilityError
from up_platform.shared.execution import OperationBackend


class TrainingBackend(OperationBackend):
    def grad_scaler(self, **kwargs):
        runtime = self.runtime
        override = getattr(runtime, "grad_scaler", None)
        if override is not None and not getattr(override, "_platform_compat", False):
            return override(**kwargs)
        if kwargs.get("enabled", True) and not runtime.is_available():
            raise CapabilityError("{} GradScaler requires an available device".format(runtime.name))
        if runtime.device_type not in ("cuda", "cpu") and hasattr(runtime.api, "amp"):
            return runtime.api.amp.GradScaler(**kwargs)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler(runtime.device_type, **kwargs)
        if runtime.api is not None and hasattr(runtime.api, "amp"):
            return runtime.api.amp.GradScaler(**kwargs)
        raise CapabilityError("{} does not provide GradScaler".format(runtime.name))

    def wrap_model(self, model, device_ids=None, **kwargs):
        return self.call("distributed.wrap_model", model, device_ids=device_ids, **kwargs)

    def wrap_legacy_model(self, model, sync=True, master_rank=0):
        from up_platform.training.legacy import wrap_model
        return wrap_model(model, sync=sync, master_rank=master_rank)

    def reduce_gradients(self, model, sync=True, allow_dead_parameter=False):
        from up_platform.training.legacy import reduce_gradients
        return reduce_gradients(model, sync=sync, allow_dead_parameter=allow_dead_parameter)

    def build_fused_optimizer(self, config):
        return self.call("build_fused_optimizer", config)

    def reload_fused_optimizer(self, optimizer):
        return self.call("reload_fused_optimizer", optimizer)
