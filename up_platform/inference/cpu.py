"""cpu inference: explicit reuse of vendor PyTorch, not the other purpose."""
from .torch_impl import TorchInference


class CpuInference(TorchInference):
    autocast = TorchInference.autocast
    export_onnx = TorchInference.export_onnx
