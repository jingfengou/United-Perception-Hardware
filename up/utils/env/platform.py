"""UP configuration stays in UP; the platform package has no YAML/CLI dependency."""
from up_platform import configure_platform
from contextlib import ExitStack
from functools import wraps


def training_validation(function):
    """Restore model/EMA modes around Runner evaluation, including exceptions."""
    @wraps(function)
    def wrapped(runner, *args, **kwargs):
        from up_platform import get_platform
        platform = get_platform()
        if platform.purpose != 'training':
            raise RuntimeError('Runner validation uses the training platform, not deployment inference')
        with ExitStack() as stack:
            stack.enter_context(platform.execution.validation(runner.model))
            if getattr(runner, 'ema', None) is not None:
                stack.enter_context(platform.execution.validation(runner.ema.model))
            return function(runner, *args, **kwargs)
    return wrapped


def configure_from_up(config, *, purpose):
    runtime = config.get('runtime', {})
    selection = dict(runtime.get('platform', {}))
    if 'device' in runtime:
        selection.setdefault('device', runtime['device'])
    if selection.get('device') == 'cpu':
        selection.setdefault('backend', 'cpu')
    return configure_platform(selection, purpose=purpose)


def configure_command(args, *, purpose):
    from up.utils.general.yaml_loader import load_yaml
    from up.utils.general.cfg_helper import merge_opts_into_cfg
    cfg = load_yaml(args.config, getattr(args, 'cfg_type', 'up'))
    cfg = merge_opts_into_cfg(getattr(args, 'opts', None), cfg)
    context = configure_from_up(cfg, purpose=purpose)
    args.platform_config = dict(cfg.get('runtime', {}).get('platform', {}))
    args.platform_config.setdefault('backend', context.vendor)
    if 'device' in cfg.get('runtime', {}):
        args.platform_config.setdefault('device', cfg['runtime']['device'])
    args.platform_purpose = purpose
    return context
