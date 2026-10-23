"""Run a frozen, same-vendor compatibility cohort without overwriting evidence.

Not an accuracy benchmark. No missing-dependency skips are counted as passes.
Run on already allocated devices; this tool does not allocate other users' GPUs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def source_manifest(root):
    files = {}
    for folder in ('up', 'up_platform', 'configs', 'tools', 'tests/platform'):
        for path in sorted((root/folder).rglob('*')):
            if path.suffix not in ('.py', '.cu', '.cpp', '.h', '.yaml', '.json') or '__pycache__' in path.parts:
                continue
            raw = path.read_bytes().replace(b'\r\n', b'\n')
            files[path.relative_to(root).as_posix()] = hashlib.sha256(raw).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'normalization': 'CRLF to LF only', 'sha256': digest, 'files': files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--vendor', choices=('nvidia', 'metax'), required=True)
    parser.add_argument('--devices', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phases', nargs='+', choices=('native', 'models', 'toolchains', 'distributed', 'cpu'),
                        default=['cpu', 'native', 'toolchains', 'distributed', 'models'])
    parser.add_argument('--models', nargs='+', default=['resnet18', 'yolov5s', 'vit_b16', 'faster_rcnn', 'pointpillar'])
    args = parser.parse_args()
    args.root, args.output, args.baseline = args.root.resolve(), args.output.resolve(), args.baseline.resolve()
    devices = args.devices.split(',')
    if not devices or not all(value.isdigit() for value in devices) or len(devices) != len(set(devices)):
        parser.error('--devices must be unique allocated GPU indices')
    if 'distributed' in args.phases and len(devices) != 2:
        parser.error('The distributed contract test requires exactly two allocated devices')
    args.output.mkdir(parents=True, exist_ok=False)
    source = source_manifest(args.root)
    (args.output/'source.json').write_text(json.dumps(source, indent=2))
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONHASHSEED='0',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', DEFAULT_TASKS='cls:det')
    env['PYTHONPATH'] = str(args.root)+os.pathsep+env.get('PYTHONPATH', '')
    records = []
    def run(name, command, visibility):
        started = datetime.now(timezone.utc).isoformat()
        with (args.output/(name+'.log')).open('w') as log:
            result = subprocess.run([sys.executable, *map(str, command)], env=dict(env, CUDA_VISIBLE_DEVICES=visibility),
                                    cwd=args.root, stdout=log, stderr=subprocess.STDOUT)
        records.append({'name': name, 'command': list(map(str, command)), 'visible_devices': visibility,
                        'exit_code': result.returncode, 'started_utc': started,
                        'ended_utc': datetime.now(timezone.utc).isoformat()})
        (args.output/'suite.json').write_text(json.dumps({'vendor': args.vendor, 'source_sha256': source['sha256'],
            'numerics': 'native', 'scope': 'compatibility, not dataset accuracy', 'records': records}, indent=2))
        print(name, result.returncode, flush=True)
    for phase in args.phases:
        if phase == 'cpu':
            run('cpu', ['-m', 'pytest', 'tests/platform', '-q', '--junitxml='+str(args.output/'cpu.xml')], '')
        elif phase == 'native':
            for purpose in ('training', 'inference'):
                command = ['tools/compare_platform_native.py', '--vendor', args.vendor, '--purpose', purpose,
                           '--baseline', args.baseline, '--output', args.output/('native-'+purpose+'.json')]
                if args.vendor == 'metax':
                    command.append('--reference-psroi-layout-fix')
                run('native-'+purpose, command, devices[0])
        elif phase == 'models':
            command = ['tools/compare_platform_models.py', '--root', args.root, '--baseline', args.baseline,
                       '--vendor', args.vendor, '--deterministic', '--models', *args.models,
                       '--output', args.output/'models']
            if args.vendor == 'metax':
                command.append('--reference-defer-kitti')
            run('models', command, devices[0])
        elif phase == 'toolchains':
            run('toolchains', ['tools/compare_platform_toolchains.py', '--vendor', args.vendor,
                              '--output', args.output/'toolchains'], devices[0])
        elif phase == 'distributed':
            run('distributed', ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                                 'tools/validate_platform_distributed.py', '--vendor', args.vendor,
                                 '--output', args.output/'distributed'], args.devices)
    final_source = source_manifest(args.root)
    assert source == final_source, 'Source changed during the cohort; this evidence cannot certify a frozen version'
    raise SystemExit(0 if all(r['exit_code'] == 0 for r in records) else 1)


if __name__ == '__main__':
    main()
