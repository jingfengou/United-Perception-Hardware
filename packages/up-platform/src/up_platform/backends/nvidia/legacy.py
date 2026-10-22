"""Optional NVIDIA/linklink implementations; imported only when requested."""
import copy


def reload_fused_optimizer(optimizer):
    for index, group in enumerate(optimizer.param_groups):
        fp16 = copy.copy(optimizer.fp16_groups[index])
        fp32 = copy.copy(optimizer.fp32_groups[index])
        masters = copy.copy(optimizer.fp32_from_fp16_groups[index])
        for position, parameter in enumerate(group["params"]):
            if parameter.requires_grad:
                if fp32 and fp32[0] is parameter:
                    fp32.pop(0)
                else:
                    master, model = masters.pop(0), fp16.pop(0)
                    group["params"][position] = model
                    if master in optimizer.state:
                        optimizer.state[model] = optimizer.state.pop(master)
    optimizer._promote_fp16_params()


def build_fused_optimizer(config):
    from up_platform import get_platform as get_accelerator
    cls = getattr(get_accelerator().dependency("link").optim, config["type"])
    cls.reload = reload_fused_optimizer
    return cls(**config["kwargs"])


def register_float_modules(modules, initialize=False):
    from up_platform import get_platform as get_accelerator
    fp16 = get_accelerator().dependency("link").fp16
    for module in modules:
        fp16.register_float_module(module, cast_args=True)
    if initialize:
        fp16.init()


def link_call(name, *args, **kwargs):
    from up_platform import get_platform as get_accelerator
    return getattr(get_accelerator().dependency("link"), name)(*args, **kwargs)


def quantization_backend_types():
    from up_platform.shared.layers import OptionalModule
    backend = OptionalModule("quantization", "prepare_by_platform").BackendType
    return {"tensorrt": backend.Tensorrt, "snpe": backend.SNPE,
            "vitis": backend.Vitis, "academic": backend.Academic, "tengine": backend.Tengine_u8}
