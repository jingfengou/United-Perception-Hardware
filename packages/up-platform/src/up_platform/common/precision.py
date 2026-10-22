"""The same autocast contract is usable by both execution purposes."""
import contextlib

import torch

from up_platform.errors import CapabilityError


def autocast(runtime, enabled=True, **kwargs):
    if enabled and runtime._numerics is not None:
        raise CapabilityError("portable_v1 does not support autocast; use FP32 model boundaries")
    if enabled and kwargs.get("dtype") not in (None, torch.float16, torch.bfloat16):
        raise CapabilityError("{} autocast does not support {}".format(runtime.name, kwargs["dtype"]))
    if not enabled and runtime.device_type == "cpu":
        return contextlib.ExitStack()
    if enabled and not runtime.is_available():
        raise CapabilityError("{} autocast requires an available device".format(runtime.name))
    if enabled and kwargs.get("dtype") is torch.bfloat16 and runtime.api is not None:
        check = getattr(runtime.api, "is_bf16_supported", None)
        if check is not None and not check():
            raise CapabilityError("{} device does not support bfloat16 autocast".format(runtime.name))
    if runtime.device_type not in ("cuda", "cpu") and hasattr(runtime.api, "amp"):
        return runtime.api.amp.autocast(enabled=enabled, **kwargs)
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(runtime.device_type, enabled=enabled, **kwargs)
    if runtime.api is not None and hasattr(runtime.api, "amp"):
        return runtime.api.amp.autocast(enabled=enabled, **kwargs)
    raise CapabilityError("{} does not provide autocast".format(runtime.name))
