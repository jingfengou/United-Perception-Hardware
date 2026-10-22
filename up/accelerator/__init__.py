"""UP configuration and legacy API facade over the independent up-platform package."""
import os

import up_platform
from up_platform import BackendConfigurationError, CapabilityError

from .registry import register_backend, register_op
from .runtime import TorchAccelerator

_active = None


class _CompatibilityContext:
    def __init__(self, context):
        self.context = context

    def __getattr__(self, name):
        return getattr(self.context, name)

    def call(self, name, *args, **kwargs):
        conventions = None
        if name in ("naive_nms", "roi_align"):
            from up.flags import ALIGNED_FLAG
            conventions = {"offset": ALIGNED_FLAG.offset, "aligned": ALIGNED_FLAG.aligned}
        return self.context.call(name, *args, _conventions=conventions, **kwargs)


def configure_accelerator(config=None, backend=None, device=None, reset=False, *, purpose=None):
    global _active
    runtime = (config or {}).get("runtime", config or {})
    options = runtime.get("accelerator", {})
    options = {"backend": options} if isinstance(options, str) else dict(options)
    options["backend"] = backend or options.get("backend") or os.environ.get("UP_ACCELERATOR_BACKEND", "auto")
    options["device"] = device if device is not None else runtime.get("device")
    if purpose is None:
        try:
            purpose = up_platform.get_platform().purpose
        except BackendConfigurationError:
            purpose = "training"
    context = up_platform.configure_platform(options, purpose=purpose, reset=reset)
    if _active is None or _active.context is not context:
        _active = _CompatibilityContext(context)
    return _active


def get_accelerator():
    global _active
    try:
        context = up_platform.get_platform()
    except BackendConfigurationError:
        return configure_accelerator()
    if _active is None or _active.context is not context:
        _active = _CompatibilityContext(context)
    return _active


def capabilities():
    return get_accelerator().capabilities()


def prepare_args(args, *, purpose=None):
    """Parse UP YAML in the parent process, without initializing a device."""
    if purpose is not None:
        args._platform_purpose = purpose
    if not hasattr(args, "_accelerator_config"):
        from up.utils.general.yaml_loader import load_yaml
        from up.utils.general.cfg_helper import merge_opts_into_cfg
        config = load_yaml(args.config, getattr(args, "cfg_type", "up"))
        config = merge_opts_into_cfg(getattr(args, "opts", None), config)
        args._accelerator_config = {"runtime": config.get("runtime", {})}
    return args._accelerator_config


def activate_args(args, *, purpose=None):
    if purpose is not None:
        prepare_args(args, purpose=purpose)
    return configure_accelerator(prepare_args(args), purpose=getattr(args, "_platform_purpose", "training"))


from . import distributed, layers, ops, sparse  # noqa: E402,F401
