from up_platform.shared.execution import OperationBackend


class InferenceBackend(OperationBackend):
    """Native forward/export routing; no implicit quantization or weight packing."""

    def export_onnx(self, model, args, destination, **kwargs):
        import inspect
        import torch
        # Keep the existing autograd symbolic path across PyTorch default changes.
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            kwargs.setdefault("dynamo", False)
        return torch.onnx.export(model, args, destination, **kwargs)
