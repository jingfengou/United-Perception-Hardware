"""One writer per output tile, fixed K order, GPU FP32 accumulation only."""
import torch

from up_platform.errors import CapabilityError


def validate_operands(a, b):
    if (a.device.type != "cuda" or b.device != a.device or a.dtype != torch.float32
            or b.dtype != torch.float32 or a.layout != torch.strided or b.layout != torch.strided
            or a.ndim != 2 or b.ndim != 2):
        raise CapabilityError("metax_deterministic_v2 mm requires dense 2D GPU FP32 operands on one device")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible matrix shapes")


def matrix(a, b):
    validate_operands(a, b)
    # Triton is optional until this explicit vendor policy executes a GPU mm.
    from .fixed_gemm_kernel import launch
    return launch(a, b)
