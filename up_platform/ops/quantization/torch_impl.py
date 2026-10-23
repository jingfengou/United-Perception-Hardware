from up_platform.dependencies import dependency


def backend_types():
    return dependency('mqbench.prepare_by_platform').BackendType


def module_type(name):
    paths = {'FrozenBatchNorm2d': 'mqbench.nn.modules', 'ConvFreezebn2d': 'mqbench.nn.intrinsic.qat.modules', 'ConvFreezebnReLU2d': 'mqbench.nn.intrinsic.qat.modules', 'FakeQuantizeBase': 'mqbench.fake_quantize.quantize_base'}
    if name not in paths:
        raise ValueError('Not a declared UP MQBench type: ' + name)
    return getattr(dependency(paths[name]), name)


def prepare(model, backend_type, prepare_custom_config_dict=None):
    config = {} if prepare_custom_config_dict is None else prepare_custom_config_dict
    return dependency('mqbench.prepare_by_platform').prepare_by_platform(model, backend_type, config)


def enable_calibration(model, quantizer_type='fake_quant'):
    return dependency('mqbench.utils.state').enable_calibration_woquantization(model, quantizer_type)


def enable_quantization(model, weight_cali_on=False, act_cali_on=False):
    return dependency('mqbench.utils.state').enable_quantization(model, weight_cali_on, act_cali_on)


def reconstruct(model, cali_data, config, graph_module_list=None):
    return dependency('mqbench.advanced_ptq').ptq_reconstruction(model, cali_data, config, graph_module_list)


def convert_deploy(model, backend_type, **kwargs):
    return dependency('mqbench.convert_deploy').convert_deploy(model, backend_type, **kwargs)

