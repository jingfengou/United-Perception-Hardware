"""ascend: explicit quantization bindings; registration is not validation."""
from .base import QuantizationOps
from up_platform.contracts import UnsupportedOperation


class AscendQuantizationOps(QuantizationOps):

    @staticmethod
    def backend_types():
        raise UnsupportedOperation("ascend", "common", "quantization.backend_types", "Vendor template: implementation and hardware validation required")
    backend_types.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def module_type(name):
        raise UnsupportedOperation("ascend", "common", "quantization.module_type", "Vendor template: implementation and hardware validation required")
    module_type.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def prepare(model, backend_type, prepare_custom_config_dict=None):
        raise UnsupportedOperation("ascend", "common", "quantization.prepare", "Vendor template: implementation and hardware validation required")
    prepare.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def enable_calibration(model, quantizer_type='fake_quant'):
        raise UnsupportedOperation("ascend", "common", "quantization.enable_calibration", "Vendor template: implementation and hardware validation required")
    enable_calibration.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def enable_quantization(model, weight_cali_on=False, act_cali_on=False):
        raise UnsupportedOperation("ascend", "common", "quantization.enable_quantization", "Vendor template: implementation and hardware validation required")
    enable_quantization.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def reconstruct(model, cali_data, config, graph_module_list=None):
        raise UnsupportedOperation("ascend", "common", "quantization.reconstruct", "Vendor template: implementation and hardware validation required")
    reconstruct.__func__.unsupported_reason = "Unimplemented vendor template"

    @staticmethod
    def convert_deploy(model, backend_type, **kwargs):
        raise UnsupportedOperation("ascend", "common", "quantization.convert_deploy", "Vendor template: implementation and hardware validation required")
    convert_deploy.__func__.unsupported_reason = "Unimplemented vendor template"

