"""Apply version-pinned import/export compatibility edits to task-local SDK sources.

No weights, masks, quantization rules, model graphs, or arithmetic are changed.
Both direct-reference and routed tests use the same patched dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PINS = {'MQBench': '00f477d58ef67c1c1bbe71d14312731fd3a4960d',
        'msbench': 'b00632036c8a03cacff6ae94814195e98f1361f8'}


def select_msbench_dist(root):
    path = root/'msbench/msbench/scheduler/amba_scheduler.py'
    receipt = root/'dependency-msbench-dist.json'
    before = path.read_bytes()
    if receipt.exists():
        assert hashlib.sha256(before).hexdigest() == json.loads(receipt.read_text())['after']
        return
    # This upstream revision refers to an undefined flag. Select its existing
    # torch.distributed branch explicitly; this does not implement linklink.
    assert b'if USE_LINKLINK:' in before and b'USE_LINKLINK =' not in before
    path.write_bytes(b'USE_LINKLINK = False  # Explicit task selection: torch.distributed, not linklink validation.\n'+before)
    receipt.write_text(json.dumps({'path': path.relative_to(root).as_posix(), 'communication': 'torch.distributed',
        'scope': 'Define missing selector for the pre-existing dist branch; pruning arithmetic unchanged',
        'before': hashlib.sha256(before).hexdigest(), 'after': hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Contains pinned MQBench and msbench source checkouts')
    parser.add_argument('--msbench-dist', choices=('torch',), help='Explicitly select the existing torch.distributed branch for Amba; never claims linklink support')
    args = parser.parse_args()
    receipt = args.root/'dependency-compatibility-v2.json'
    if receipt.exists():
        for item in json.loads(receipt.read_text())['patches']:
            assert hashlib.sha256((args.root/item['path']).read_bytes()).hexdigest() == item['after'], item['path']
        print('Existing compatibility edits verified')
        if args.msbench_dist:
            select_msbench_dist(args.root)
        return
    records = []
    # Exported source archives are also accepted if their Git revision manifest is supplied.
    for package, expected in PINS.items():
        root = args.root/package
        if (root/'.git').exists():
            revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        else:
            revision = json.loads((args.root/'dependency-pins.json').read_text())[package]
        if revision != expected:
            raise RuntimeError(f'{package}: require pinned {expected}, got {revision}')
    for path in (args.root/'MQBench/mqbench').rglob('*.py'):
        before = path.read_bytes()
        source = before.decode().replace('\r\n', '\n')
        changed = source.replace('from torch.onnx import _type_utils',
                                 'from torch.onnx._internal.torchscript_exporter import _type_utils')
        if '    _type_utils,\n' in changed:
            changed = changed.replace('    _type_utils,\n', '')
            changed = 'from torch.onnx._internal.torchscript_exporter import _type_utils\n' + changed
        changed = changed.replace('from torch.onnx._internal import jit_utils, registration',
                                  'from torch.onnx._internal.torchscript_exporter import jit_utils, registration')
        if path.name == 'convert_deploy.py':
            changed = changed.replace('torch.onnx.export(model, dummy_input, onnx_model_path,\n',
                                      'torch.onnx.export(model, dummy_input, onnx_model_path, dynamo=False,\n')
        if changed != source:
            path.write_text(changed, newline='\n')
            records.append({'path': path.relative_to(args.root).as_posix(),
                            'before': hashlib.sha256(before).hexdigest(), 'after': hashlib.sha256(path.read_bytes()).hexdigest()})
    path = args.root/'msbench/msbench/fake_sparse/__init__.py'
    before = path.read_bytes()
    # The class exists in this revision but is missing from its public import list.
    if b'FCPTSFakeSparse' not in before:
        path.write_bytes(before + b'\nfrom .sparse_base import FCPTSFakeSparse\n')
        records.append({'path': path.relative_to(args.root).as_posix(), 'before': hashlib.sha256(before).hexdigest(),
                        'after': hashlib.sha256(path.read_bytes()).hexdigest()})
    path = args.root/'msbench/msbench/scheduler/nv_asp_cores/asp.py'
    before = path.read_bytes()
    changed = before.replace(b"torchvision.ops.misc.Conv2d: ['weight']",
                             b"getattr(torchvision.ops.misc, 'Conv2d', torch.nn.Conv2d): ['weight']")
    if changed != before:
        path.write_bytes(changed)
        records.append({'path': path.relative_to(args.root).as_posix(), 'before': hashlib.sha256(before).hexdigest(),
                        'after': hashlib.sha256(changed).hexdigest()})
    receipt.write_text(json.dumps({'pins': PINS, 'torch': '2.10', 'scope': 'imports and legacy ONNX exporter only',
                                   'patches': records}, indent=2)+'\n')
    print(receipt)
    if args.msbench_dist:
        select_msbench_dist(args.root)


if __name__ == '__main__':
    main()
