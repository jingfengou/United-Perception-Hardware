import torch

from .base import InferenceBackend


class TorchInference(InferenceBackend):
    def __init__(self, device):
        self.device = device

    def autocast(self, enabled=True, dtype=None, cache_enabled=True):
        return torch.amp.autocast(self.device.device_type, enabled=enabled, dtype=dtype, cache_enabled=cache_enabled)

    def export_onnx(self, model, args, destination, **kwargs):
        # UP uses autograd.Function.symbolic; PyTorch 2.9 changed the default.
        import inspect
        if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
            kwargs.setdefault('dynamo', False)
        return torch.onnx.export(model, args, destination, **kwargs)
