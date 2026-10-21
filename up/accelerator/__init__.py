"""Hardware-neutral entry point. Importing this package does not load a GPU SDK."""
import importlib
import os

from .errors import BackendConfigurationError, CapabilityError
from .registry import BACKENDS, BACKEND_PROBES, register_backend, register_op, resolve
from .runtime import TorchAccelerator

_active = None
_selected_name = None

register_backend("cpu", TorchAccelerator)
for _name, _class in (
        ("nvidia", "NvidiaAccelerator"), ("ascend", "AscendAccelerator"),
        ("cambricon", "CambriconAccelerator"), ("metax", "MetaXAccelerator")):
    register_backend(_name, "up.accelerator.backends.torch_backends:" + _class)


def configure_accelerator(config=None, backend=None, device=None, reset=False):
    global _active, _selected_name
    runtime = (config or {}).get("runtime", config or {})
    options = runtime.get("accelerator", {})
    if isinstance(options, str):
        options = {"backend": options}
    for plugin in options.get("plugins", []):
        importlib.import_module(plugin)
    name = backend or options.get("backend") or os.environ.get("UP_ACCELERATOR_BACKEND", "auto")
    target = device if device is not None else runtime.get("device")
    if name == "auto":
        if target is not None and str(target).split(":")[0] == "cpu":
            name = "cpu"
        else:
            matches = [candidate for candidate, probe in BACKEND_PROBES.items() if probe()]
            if len(matches) > 1:
                raise BackendConfigurationError("Ambiguous accelerator probes: " + ", ".join(matches))
            if matches:
                name = matches[0]
    if name == "auto":
        # Native device namespaces identify extension backends. CUDA-shaped
        # vendor distributions must select their backend explicitly.
        import torch
        if hasattr(torch, "npu") or hasattr(torch, "mlu"):
            raise BackendConfigurationError("Select ascend/cambricon explicitly in runtime.accelerator.backend")
        if getattr(torch.version, "maca", None) or getattr(torch.version, "metax", None):
            raise BackendConfigurationError("Select metax explicitly; a cuda device is not a vendor identity")
        name = "nvidia" if torch.version.cuda else "cpu"
    if name not in BACKENDS:
        raise BackendConfigurationError("Unknown accelerator {!r}; available: {}".format(name, sorted(BACKENDS)))
    if _active is not None and not reset:
        if name != _selected_name:
            raise BackendConfigurationError("Backend already configured as {}; cannot switch to {}".format(
                _selected_name, name))
        if target is not None and str(target).split(":")[0] != _active.device_type:
            raise BackendConfigurationError("Configured backend/device mismatch: " + str(target))
        _active.validate_environment(options.get("requirements", {}))
        if "numerics" in options:
            _active._configure_numerics(options["numerics"])
        return _active
    try:
        instance = resolve(BACKENDS[name])(device=target)
    except (ImportError, OSError) as exc:
        raise BackendConfigurationError("Cannot initialize {}: {}".format(name, exc)) from exc
    instance.name = name
    instance.validate_environment(options.get("requirements", {}))
    if _active is not None:
        _active._close_numerics()
        _active, _selected_name = None, None
    instance._configure_numerics(options.get("numerics", "native"))
    _active, _selected_name = instance, name
    return instance


def get_accelerator():
    return _active if _active is not None else configure_accelerator()


def capabilities():
    return get_accelerator().capabilities()


def prepare_args(args):
    """Read effective configuration in the parent without initializing a device."""
    if not hasattr(args, "_accelerator_config"):
        from up.utils.general.yaml_loader import load_yaml
        from up.utils.general.cfg_helper import merge_opts_into_cfg
        config = load_yaml(args.config, getattr(args, "cfg_type", "up"))
        config = merge_opts_into_cfg(getattr(args, "opts", None), config)
        args._accelerator_config = {"runtime": config.get("runtime", {})}
    return args._accelerator_config


def activate_args(args):
    return configure_accelerator(prepare_args(args))


from . import distributed, layers, ops, sparse  # noqa: E402,F401
