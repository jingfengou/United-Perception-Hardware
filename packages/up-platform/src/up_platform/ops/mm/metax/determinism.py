"""Opt-in C500 FP32 convolution reduction plus deterministic 2D GEMM."""
import torch

from up_platform.errors import CapabilityError

from up_platform.ops.convolution.metax.weight_gradient import DeterministicConvolution

PROFILE = "metax_deterministic_v2"


class DeterministicGemm(DeterministicConvolution):
    profile = PROFILE

    def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if operation._schema.name == "aten::mm":
            operands = dict(zip(("self", "mat2"), args))
            operands.update(kwargs)
            a, b = operands["self"], operands["mat2"]
            if a.device.type == "cuda" or b.device.type == "cuda":
                if operation is not torch.ops.aten.mm.default:
                    raise CapabilityError(PROFILE + " supports aten.mm.default, not " + str(operation))
                from .fixed_gemm import matrix
                result = matrix(a, b)
                key = "aten.mm.default"
                self.counts[key] = self.counts.get(key, 0) + 1
                return result
        return super().__torch_dispatch__(operation, types, args, kwargs)
