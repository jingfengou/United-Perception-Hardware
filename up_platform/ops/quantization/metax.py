"""metax: explicit quantization bindings; registration is not validation."""
from .base import QuantizationOps
from . import torch_impl


class MetaxQuantizationOps(QuantizationOps):
    dependencies = ("mqbench",)
    backend_types = staticmethod(torch_impl.backend_types)
    module_type = staticmethod(torch_impl.module_type)
    prepare = staticmethod(torch_impl.prepare)
    enable_calibration = staticmethod(torch_impl.enable_calibration)
    enable_quantization = staticmethod(torch_impl.enable_quantization)
    reconstruct = staticmethod(torch_impl.reconstruct)
    convert_deploy = staticmethod(torch_impl.convert_deploy)

