from abc import ABC, abstractmethod
from up_platform.contracts import operation


class QuantizationOps(ABC):

    @staticmethod
    @operation(source="quantization.backend_types", output="MQBench BackendType enum; deployment target remains an explicit UP choice.", gradients="None",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def backend_types():
        "MQBench BackendType enum; deployment target remains an explicit UP choice."
        raise NotImplementedError

    @staticmethod
    @operation(source="quantization.module_type", output="Original MQBench class for construction or isinstance checks.", gradients="Class-dependent",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def module_type(name):
        "Original MQBench class for construction or isinstance checks."
        raise NotImplementedError

    @staticmethod
    @operation(source="quantization.prepare", output="Original MQBench prepared model; observer/quantizer state and FX graph.", gradients="Preserve MQBench fake quantization gradients",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def prepare(model, backend_type, prepare_custom_config_dict=None):
        "Original MQBench prepared model; observer/quantizer state and FX graph."
        raise NotImplementedError

    @staticmethod
    @operation(source="quantization.enable_calibration", output="Enable the original observer state; no implicit data movement.", gradients="Original observer semantics",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def enable_calibration(model, quantizer_type='fake_quant'):
        "Enable the original observer state; no implicit data movement."
        raise NotImplementedError

    @staticmethod
    @operation(source="quantization.enable_quantization", output="Update original quantizer/observer flags in place.", gradients="Original fake-quant gradients",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def enable_quantization(model, weight_cali_on=False, act_cali_on=False):
        "Update original quantizer/observer flags in place."
        raise NotImplementedError

    @staticmethod
    @operation(source="quantization.reconstruct", output="Original MQBench PTQ result with original algorithm/config.", gradients="Original reconstruction gradients",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def reconstruct(model, cali_data, config, graph_module_list=None):
        "Original MQBench PTQ result with original algorithm/config."
        raise NotImplementedError

    @staticmethod
    @operation(source="quantization.convert_deploy", output="Original deployment artifacts; backend does not imply a runtime engine.", gradients="None; export path",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def convert_deploy(model, backend_type, **kwargs):
        "Original deployment artifacts; backend does not imply a runtime engine."
        raise NotImplementedError

