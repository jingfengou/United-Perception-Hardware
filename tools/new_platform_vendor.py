"""Generate explicit vendor files from the current abstract interfaces.

Output is a standalone plugin directory, never an edit of the existing vendors.
--reuse-torch is an explicit compatibility declaration, not automatic detection.
"""
import argparse
import importlib
import inspect
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TORCH_CATEGORIES = {'matrix', 'convolution', 'normalization', 'activation', 'pooling', 'loss', 'attention'}


def generate(vendor, device_type, output, reuse_torch=False):
    if not re.fullmatch('[a-z][a-z0-9_]*', vendor):
        raise ValueError('vendor must be a lowercase Python identifier')
    if not re.fullmatch('[a-z][a-z0-9_]*', device_type):
        raise ValueError('device type must be a lowercase identifier')
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    bindings = []
    for path in sorted((ROOT/'up_platform/ops').glob('*/base.py')):
        category = path.parent.name
        module = importlib.import_module('up_platform.ops.'+category+'.base')
        base = next(value for name, value in vars(module).items() if name.endswith('Ops'))
        cls = vendor.title().replace('_', '')+base.__name__
        lines = [f'from up_platform.ops.{category}.base import {base.__name__}',
                 'from up_platform.contracts import UnsupportedOperation']
        shared = reuse_torch and category in TORCH_CATEGORIES
        if shared:
            lines.append(f'from up_platform.ops.{category} import torch_impl')
        lines += ['', '', f'class {cls}({base.__name__}):', f'    dependencies = {("torch",) if shared else ()!r}']
        for name in sorted(base.__abstractmethods__):
            if shared:
                lines.append(f'    {name} = staticmethod(torch_impl.{name})')
            else:
                signature = inspect.signature(getattr(base, name))
                lines += ['    @staticmethod', f'    def {name}{signature}:',
                          f'        raise UnsupportedOperation({vendor!r}, "common", "{category}.{name}", "Unimplemented vendor template")',
                          f'    {name}.__func__.unsupported_reason = "Unimplemented vendor template"', '']
        files['ops/'+category+'.py'] = '\n'.join(lines)+'\n'
        bindings.append((category, cls))
    for purpose in ('training', 'inference'):
        module = importlib.import_module('up_platform.'+purpose+'.base')
        base = getattr(module, purpose.title()+'Backend')
        shared_class = 'Torch'+purpose.title()
        lines = [f'from up_platform.{purpose}.base import {base.__name__}',
                 'from up_platform.contracts import UnsupportedOperation']
        if reuse_torch:
            lines.append(f'from up_platform.{purpose}.torch_impl import {shared_class}')
        lines += ['', '', f'class Execution({base.__name__}):', '    def __init__(self, device):', '        self.device = device', '']
        for name in sorted(base.__abstractmethods__):
            if reuse_torch:
                lines.append(f'    {name} = {shared_class}.{name}')
            else:
                lines += [f'    def {name}{inspect.signature(getattr(base, name))}:',
                          f'        raise UnsupportedOperation({vendor!r}, {purpose!r}, {name!r}, "Unimplemented vendor template")', '']
        files[purpose+'.py'] = '\n'.join(lines)+'\n'
    files['device.py'] = f'''from up_platform.backends.torch_device import TorchDevice


class Device(TorchDevice):
    vendor = {vendor!r}
    device_type = {device_type!r}

    def __init__(self, device=None):
        # Import your vendor SDK here, then verify its build identity/version.
        # Override any incompatible device methods, especially convolution_flags.
        super().__init__(device)
'''
    lines = ['from importlib import import_module', 'from up_platform import register_backend',
             'from up_platform.base import PlatformBackend', '', '', 'def build(config, purpose):',
             '    from .device import Device', '    device = Device(config.get("device"))', '    operations = {']
    lines += [f'        {category!r}: import_module(__name__+".ops.{category}").{cls}(),' for category, cls in bindings]
    lines += ['    }', '    execution = import_module(__name__+"."+purpose).Execution(device)',
              '    return PlatformBackend(device, purpose, operations, execution)', '', f'register_backend({vendor!r}, build)', '']
    files['__init__.py'] = '\n'.join(lines)
    files['ops/__init__.py'] = ''
    for name, content in files.items():
        compile(content, name, 'exec')
        path = output/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf8', newline='\n')
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendor', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-torch', action='store_true')
    args = parser.parse_args()
    print('Generated', len(generate(args.vendor, args.device, args.output, args.reuse_torch)), 'files; templates are NOT validated hardware support')
