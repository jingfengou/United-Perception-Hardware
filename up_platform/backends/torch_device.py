"""Shared implementation for devices whose vendor supports the PyTorch API."""
from collections.abc import Mapping

import torch

from up_platform.base import DeviceBackend
from up_platform.contracts import UnsupportedOperation


class TorchDevice(DeviceBackend):
    def __init__(self, device=None):
        selected = torch.device(device or self.device_type)
        if selected.type != self.device_type:
            raise ValueError(f"{self.vendor} uses {self.device_type}, not {selected.type}")
        self._selected = selected

    @property
    def api(self):
        return getattr(torch, self.device_type)

    def available(self):
        return self.device_type == "cpu" or self.api.is_available()

    def device(self, index=None):
        return self._selected if index is None else torch.device(self.device_type, index)

    def set_device(self, index):
        if self.device_type != "cpu":
            self.api.set_device(index)
        self._selected = self.device(index)

    def device_count(self):
        return 1 if self.device_type == "cpu" else self.api.device_count()

    def current_device(self):
        return 0 if self.device_type == "cpu" else self.api.current_device()

    def is_accelerator(self, tensor):
        return self.device_type != 'cpu' and tensor.device.type == self.device_type

    def properties(self, index=None):
        if self.device_type == "cpu":
            raise UnsupportedOperation(self.vendor, "common", "device.properties", "CPU test backend has no accelerator properties")
        return self.api.get_device_properties(index)

    def move(self, value, device=None, non_blocking=False):
        destination = self.device(device) if isinstance(device, int) else (device or self.device())
        if isinstance(value, (torch.Tensor, torch.nn.Module)):
            return value.to(device=destination, non_blocking=non_blocking)
        if isinstance(value, (torch.UntypedStorage, torch.TypedStorage)):
            return value.to(device=torch.device(destination), non_blocking=non_blocking)
        if isinstance(value, Mapping):
            return type(value)((k, self.move(v, destination, non_blocking)) for k, v in value.items())
        if isinstance(value, tuple):
            items = [self.move(v, destination, non_blocking) for v in value]
            return type(value)(*items) if hasattr(value, "_fields") else type(value)(items)
        if isinstance(value, list):
            return [self.move(v, destination, non_blocking) for v in value]
        return value

    def synchronize(self, device=None):
        if self.device_type != "cpu":
            self.api.synchronize(device)

    def empty_cache(self):
        if self.device_type != "cpu":
            self.api.empty_cache()

    def memory_allocated(self, device=None):
        return 0 if self.device_type == "cpu" else self.api.memory_allocated(device)

    def max_memory_allocated(self, device=None):
        return 0 if self.device_type == "cpu" else self.api.max_memory_allocated(device)

    def reset_peak_memory_stats(self, device=None):
        if self.device_type != "cpu":
            self.api.reset_peak_memory_stats(device)

    def get_rng_state(self, device=None):
        selected = self.device() if device is None else device
        return torch.get_rng_state() if self.device_type == "cpu" else self.api.get_rng_state(selected)

    def set_rng_state(self, state, device=None):
        if self.device_type == "cpu":
            torch.set_rng_state(state)
        else:
            self.api.set_rng_state(state, self.device() if device is None else device)

    def manual_seed(self, seed):
        if self.device_type == "cpu":
            torch.random.default_generator.manual_seed(seed)
        else:
            self.api.manual_seed(seed)

    def convolution_flags(self, *, enabled=None, benchmark=None, deterministic=None, allow_tf32=None):
        backend = torch.backends.cudnn
        for name, value in {"enabled": enabled, "benchmark": benchmark,
                            "deterministic": deterministic, "allow_tf32": allow_tf32}.items():
            if value is not None:
                setattr(backend, name, value)
        return {name: getattr(backend, name) for name in ("enabled", "benchmark", "deterministic", "allow_tf32")}

    def environment(self):
        return {"vendor": self.vendor, "device_type": self.device_type, "torch": torch.__version__,
                "cuda_api_version": torch.version.cuda}
