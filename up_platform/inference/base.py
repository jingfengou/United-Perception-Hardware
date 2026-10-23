from abc import ABC, abstractmethod


class InferenceBackend(ABC):
    @abstractmethod
    def autocast(self, enabled=True, dtype=None, cache_enabled=True):
        """Return the inference autocast context, without enabling quantization."""

    @abstractmethod
    def export_onnx(self, model, args, destination, **kwargs):
        """Run the original export with unchanged opset, symbols and named outputs."""
