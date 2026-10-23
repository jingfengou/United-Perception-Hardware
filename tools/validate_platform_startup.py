"""Check real CLI imports and original SegFormer default-module compatibility."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

import torch

from compare_platform_models import compare, cpu
from run_platform_validation import source_manifest


def attention(path):
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MultiheadAttention')
    from up_platform import layers
    namespace = {'nn': torch.nn, 'torch': torch, 'warnings': warnings, 'platform_layers': layers}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['MultiheadAttention']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--vendor', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = source_manifest(args.root)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', DEFAULT_TASKS=':'.join(sorted(
        p.name for p in (args.root/'up/tasks').iterdir() if (p/'__init__.py').is_file())))
    commands = []
    for name, command in [('cpu', ['-m', 'pytest', 'tests/platform', '-q', '--junitxml='+str(args.output/'cpu.xml')]),
                          ('root', ['-m', 'up', '--help']), ('train', ['-m', 'up', 'train', '--help']),
                          ('inference', ['-m', 'up', 'inference', '--help'])]:
        with (args.output/(name+'.log')).open('w') as log:
            run = subprocess.run([sys.executable, *command], cwd=args.root, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
        commands.append({'name': name, 'arguments': command, 'exit_code': run.returncode})
    relative = 'up/tasks/seg/models/encoder/segformer_encoder.py'
    # Define the routed class before selection: default arguments must not initialize a vendor.
    routed = attention(args.root/relative)
    original = attention(args.baseline/relative)
    from up_platform import configure_platform
    p = configure_platform({'backend': args.vendor}, purpose='training')
    values = {}
    for name, cls in [('original', original), ('repeat', original), ('routed', routed)]:
        torch.manual_seed(131)
        module = cls(8, 2, batch_first=True)
        initial = cpu(module.state_dict())
        rng = torch.get_rng_state().clone()
        # Repeated constructors retain the original shared Dropout default.
        second = cls(8, 2, batch_first=True)
        assert module.dropout_layer is second.dropout_layer
        assert type(module.dropout_layer) is torch.nn.Dropout and module.dropout_layer.p == 0
        assert isinstance(cls(8, 2, dropout_layer=None).dropout_layer, torch.nn.Identity)
        module.to(p.device.device())
        x = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(500)).to(p.device.device()).requires_grad_()
        output = module(x)
        output.square().mean().backward()
        values[name] = {'initial': initial, 'rng': rng, 'output': cpu(output), 'input_grad': cpu(x.grad),
                        'gradients': {key: cpu(value.grad) for key, value in module.named_parameters()}}
    torch.save(values, args.output/'values.pt')
    repeat = compare(values['original'], values['repeat'])
    result = compare(values['original'], values['routed'])
    assert source == source_manifest(args.root), 'Source changed during startup verification'
    report = {'commands': commands, 'environment': p.device.environment(), 'source': source,
              'original_repeat': repeat, 'routed': result, 'rtol': 1e-4, 'atol': 1e-5,
              'scope': 'All-task CLI help without GPU; original SegFormer attention constructor/default, RNG, forward and gradients on GPU',
              'passed': all(c['exit_code'] == 0 for c in commands) and repeat['passed'] and result['passed']}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'source'}))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
