"""Read-only full-tensor observations for one diagnostic training step."""
import hashlib

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes


class DispatchTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.phase = "forward"

    def describe(self, value):
        if isinstance(value, torch.Tensor):
            with _disable_current_modes():
                tensor = value.detach().cpu().contiguous()
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.view(torch.int16)
                array = tensor.numpy()
                payload = array.tobytes()
                numerical = None
                if tensor.dtype == torch.float32:
                    import numpy as np
                    canonical = np.where(array == 0, np.float32(0), array)
                    numerical = hashlib.sha256(canonical.tobytes()).hexdigest()
            result = {"shape": list(value.shape), "dtype": str(value.dtype),
                      "device_type": value.device.type,
                      "sha256": hashlib.sha256(payload).hexdigest()}
            if numerical is not None:
                result["zero_canonical_sha256"] = numerical
            return result
        if isinstance(value, (tuple, list)):
            return [self.describe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self.describe(item) for key, item in value.items()}
        if isinstance(value, (bool, int, float, str, type(None))):
            return value
        return str(value)

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        inputs = self.describe((args, kwargs))
        output = function(*args, **kwargs)
        self.rows.append({"index": len(self.rows), "phase": self.phase,
                          "operation": str(function), "inputs": inputs,
                          "output": self.describe(output)})
        return output
