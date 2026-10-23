"""Same-vendor original UP ModelHelper/loss/optimizer compatibility, not accuracy.

Three independent processes (original, original repeat, routed) use the same
CPU initialization, synthetic inputs, seed, native policy and rtol/atol.
Independent inference is a fresh process, not an eval-mode routing guess.
"""
import argparse
import collections
import collections.abc
import copy
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch

MODELS = {'resnet18': 'configs/cls/resnet/res18.yaml',
          'yolov5s': 'configs/det/yolov5/yolov5_small_silu.yaml',
          'vit_b16': 'configs/cls/vision_transformer/vit_b16_224.yaml',
          'faster_rcnn': 'configs/det/faster_rcnn/faster_rcnn_r50_fpn_1x.yaml',
          'pointpillar': 'configs/det_3d/pointpillar/pointpillar.yaml'}


class LazyOptionalReader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Defer only the unused PCD reader's Open3D import in the original reference."""
    def __init__(self, baseline):
        self.path = baseline/'up/tasks/det_3d/data/data_reader.py'

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'up.tasks.det_3d.data.data_reader':
            return importlib.util.spec_from_loader(fullname, self, origin=str(self.path))

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        source = self.path.read_text()
        assert source.count('import open3d as o3d') == 1
        source = source.replace('import open3d as o3d\n', '')
        source = source.replace('    def get_lidar_pcd(self, path):\n',
                                '    def get_lidar_pcd(self, path):\n        import open3d as o3d\n')
        module.__file__ = str(self.path)
        exec(compile(source, str(self.path), 'exec'), module.__dict__)


class LazyKittiEvaluator(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Defer an unused CUDA JIT, never substitute its calculation or output."""
    def __init__(self, baseline):
        self.path = baseline/'up/tasks/det_3d/data/metrics/kitti_object_eval_python/rotate_iou.py'

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'up.tasks.det_3d.data.metrics.kitti_object_eval_python.rotate_iou':
            return importlib.util.spec_from_loader(fullname, self, origin=str(self.path))

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        def rotate_iou_gpu_eval(*args, **kwargs):
            namespace = {'__name__': module.__name__, '__file__': str(self.path)}
            exec(compile(self.path.read_text(), str(self.path), 'exec'), namespace)
            return namespace['rotate_iou_gpu_eval'](*args, **kwargs)
        module.__file__ = str(self.path)
        module.__all__ = ['rotate_iou_gpu_eval']
        module.rotate_iou_gpu_eval = rotate_iou_gpu_eval


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(v) for v in value)
    return value


def flatten(value, prefix=''):
    if isinstance(value, np.ndarray):
        return {prefix: torch.from_numpy(value.copy())}
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if isinstance(value, dict):
        return {k: v for name, child in value.items() for k, v in flatten(child, prefix+'/'+str(name)).items()}
    if isinstance(value, (tuple, list)):
        return {k: v for i, child in enumerate(value) for k, v in flatten(child, prefix+'/'+str(i)).items()}
    return {prefix: value}


def compare(left, right):
    a, b = flatten(left), flatten(right)
    errors = []
    for key in sorted(a.keys() | b.keys()):
        if key not in a or key not in b:
            errors.append({'tensor': key, 'reason': 'missing'})
            continue
        x, y = a[key], b[key]
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            if type(x) is not type(y) or x != y:
                errors.append({'tensor': key, 'reason': 'metadata/value'})
        elif x.shape != y.shape or x.dtype != y.dtype:
            errors.append({'tensor': key, 'reason': 'shape/dtype'})
        elif not (torch.allclose(x, y, rtol=1e-4, atol=1e-5) if x.is_floating_point() else torch.equal(x, y)):
            errors.append({'tensor': key, 'max_abs': float((x.double()-y.double()).abs().max())})
    return {'passed': not errors, 'tensor_count': len(a), 'failures': errors}


def seed():
    random.seed(131)
    np.random.seed(131)
    torch.manual_seed(131)
    torch.cuda.manual_seed_all(131)


def worker(args):
    # PointPillar's registry lives in det_3d, which ordinary 2D runs need not load.
    if args.model == 'pointpillar':
        tasks = os.environ.get('DEFAULT_TASKS', 'cls:det').split(':')
        os.environ['DEFAULT_TASKS'] = ':'.join(dict.fromkeys([*tasks, 'det_3d']))
    from up_platform import configure_platform
    from up_platform.ops._native.loader import load_native
    platform = configure_platform({'backend': args.vendor}, purpose=args.purpose)
    platform.device.set_device(0)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        platform.device.convolution_flags(benchmark=False, deterministic=True)
    if args.variant != 'routed':
        # Python 3.12 removed the old collections aliases; no model math changed.
        collections.Iterable = collections.abc.Iterable
        # Removed NumPy aliases were exactly the Python builtins, not float32.
        np.float, np.int, np.bool = float, int, bool
        sys.path.insert(0, str(args.baseline))
        sys.modules['up.extensions._C'] = load_native(args.vendor)
        if args.model == 'pointpillar':
            sys.meta_path.insert(0, LazyOptionalReader(args.baseline))
            if args.reference_defer_kitti:
                sys.meta_path.insert(0, LazyKittiEvaluator(args.baseline))
    from up.models.model_helper import ModelHelper
    from up.utils.general.yaml_loader import load_yaml
    from up.utils.model.optimizer_helper import BaseOptimizer
    config_path = args.root / MODELS[args.model]
    config = load_yaml(str(config_path))
    os.chdir(config_path.parent)
    seed()
    model = ModelHelper(copy.deepcopy(config['net']))
    record = {'initial': cpu(model.state_dict()), 'parameter_names': [n for n, _ in model.named_parameters()]}
    model = platform.device.move(model) if args.variant == 'routed' else model.cuda()
    generator = torch.Generator().manual_seed(500)
    size = 128 if args.model in ('yolov5s', 'faster_rcnn') else 224
    original_batch = {'image': torch.randn(2, 3, size, size, generator=generator), 'gt': torch.tensor([1, 2]),
                      'gt_bboxes': [torch.tensor([[15., 16., 60., 65., 1.]]) for _ in range(2)],
                      'image_info': [[size, size, 1., size, size, False, 0, 0] for _ in range(2)]}
    if args.model == 'pointpillar':
        grid = np.array([432, 496, 1], dtype=np.int64)
        coords = torch.tensor([[b, 0, 100+i//16, 100+i%16] for b in range(2) for i in range(128)], dtype=torch.int32)
        voxels = torch.rand(256, 32, 4, generator=generator)
        voxels[:, :, 0] = (coords[:, 3, None]+voxels[:, :, 0])*.16
        voxels[:, :, 1] = (coords[:, 2, None]+voxels[:, :, 1])*.16-39.68
        voxels[:, :, 2] = voxels[:, :, 2]*4-3
        original_batch = {'voxels': voxels, 'voxel_coords': coords, 'voxel_num_points': torch.full((256,), 32),
                          'voxel_infos': {'grid_size': grid, 'voxel_size': [.16, .16, 4],
                                          'point_cloud_range': [0, -39.68, -3, 69.12, 39.68, 1]},
                          'gt_boxes': torch.tensor([[[16., -23., -1., 3.9, 1.6, 1.56, 0., 1.]]]*2),
                          'class_names': ['Car', 'Pedestrian', 'Cyclist'], 'batch_size': 2}
    def batch():
        return platform.device.move(copy.deepcopy(original_batch))
    if args.purpose == 'inference':
        # Isolate inference equivalence from any training nondeterminism.
        checkpoint = args.output.parent / f'{args.model}-original-training.checkpoint.pt'
        state = torch.load(checkpoint, weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        model.eval()
        with torch.no_grad():
            record = {'inference': cpu(model(batch()))}
    else:
        optimizer_cfg = copy.deepcopy(config['trainer']['optimizer'])
        optimizer = BaseOptimizer(optimizer_cfg, model).build_optimizer()
        record['steps'] = []
        for step in range(3):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(batch())
            losses = {k: v for k, v in output.items() if 'loss' in k and isinstance(v, torch.Tensor)}
            if not losses:
                raise AssertionError('Expected actual UP model loss, not a synthetic feature loss')
            sum(losses.values()).backward()
            record['steps'].append({'losses': cpu(losses), 'gradients': {k: cpu(p.grad) for k, p in model.named_parameters()}})
            optimizer.step()
            before = cpu(model.state_dict())
            modes = [m.training for m in model.modules()]
            if args.variant == 'routed':
                with platform.execution.validation(model):
                    evaluated = cpu(model(batch()))
                assert platform.purpose == 'training'
            else:
                model.eval()
                with torch.no_grad():
                    evaluated = cpu(model(batch()))
                for child, mode in zip(model.modules(), modes):
                    child.training = mode
            assert [m.training for m in model.modules()] == modes
            assert all(torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items())
            record['steps'][-1]['eval'] = evaluated
            if step == 1:
                saved = cpu({'model': model.state_dict(), 'optimizer': optimizer.state_dict()})
                rng = (torch.get_rng_state(), platform.device.get_rng_state(), np.random.get_state(), random.getstate())
                # A fresh model and optimizer exercise actual state restoration.
                model = ModelHelper(copy.deepcopy(config['net']))
                model = platform.device.move(model) if args.variant == 'routed' else model.cuda()
                model.load_state_dict(saved['model'], strict=True)
                optimizer = BaseOptimizer(copy.deepcopy(optimizer_cfg), model).build_optimizer()
                optimizer.load_state_dict(saved['optimizer'])
                # Constructor RNG draws must not perturb the continuation stream.
                torch.set_rng_state(rng[0])
                platform.device.set_rng_state(rng[1])
                np.random.set_state(rng[2])
                random.setstate(rng[3])
        record['final'] = cpu(model.state_dict())
        record['optimizer'] = cpu(optimizer.state_dict())
        torch.save({'model': record['final'], 'optimizer': record['optimizer']}, args.output.with_suffix('.checkpoint.pt'))
    torch.save(record, args.output.with_suffix('.pt'))
    args.output.write_text(json.dumps({'vendor': args.vendor, 'variant': args.variant, 'model': args.model,
                                      'purpose': args.purpose, 'passed_execution': True,
                                      'tasks': os.environ.get('DEFAULT_TASKS'),
                                      'config': MODELS[args.model], 'seed': 131, 'numerics': 'native',
                                      'deterministic_algorithms': args.deterministic,
                                      'reference_adaptations': ['collections.Iterable Python 3.12 alias', 'NumPy removed float/int/bool builtin aliases'] +
                                          (['defer unused Open3D PCD reader import'] if args.model == 'pointpillar' else []) +
                                          (['defer unused KITTI Numba CUDA JIT import; original calculation unchanged if called']
                                           if args.model == 'pointpillar' and args.reference_defer_kitti else []),
                                      'source': str(Path(sys.modules['up'].__file__).resolve()),
                                      'environment': platform.device.environment()}, indent=2))


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for model in args.models:
        for purpose in ('training', 'inference'):
            paths = {}
            for variant in ('original', 'repeat', 'routed'):
                path = args.output / f'{model}-{variant}-{purpose}.json'
                if path.exists() or path.with_suffix('.log').exists():
                    raise FileExistsError('Use a new evidence directory; do not overwrite trials: '+str(path))
                command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--root', str(args.root),
                           '--baseline', str(args.baseline), '--vendor', args.vendor, '--model', model,
                           '--variant', variant, '--purpose', purpose, '--output', str(path)]
                if args.deterministic:
                    command.append('--deterministic')
                if args.reference_defer_kitti:
                    command.append('--reference-defer-kitti')
                with path.with_suffix('.log').open('w') as log:
                    run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                paths[variant] = path
                print(model, purpose, variant, run.returncode, flush=True)
            record = {'model': model, 'purpose': purpose}
            if all(p.exists() for p in paths.values()):
                artifacts = {k: torch.load(p.with_suffix('.pt'), weights_only=False) for k, p in paths.items()}
                record['original_repeat'] = compare(artifacts['original'], artifacts['repeat'])
                record['routed'] = compare(artifacts['original'], artifacts['routed'])
                record['artifact_sha256'] = {k: hashlib.file_digest(p.with_suffix('.pt').open('rb'), 'sha256').hexdigest() for k, p in paths.items()}
            else:
                record['failed_execution'] = [k for k, p in paths.items() if not p.exists()]
            results.append(record)
            (args.output / 'summary.json').write_text(json.dumps({'scope': 'UP ModelHelper, actual losses, three updates with checkpoint restore; synthetic inputs, not full Runner or accuracy', 'rtol': 1e-4, 'atol': 1e-5, 'records': results}, indent=2))
    raise SystemExit(0 if all(r.get('routed', {}).get('passed', False) and
                             r.get('original_repeat', {}).get('passed', False) for r in results) else 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--vendor', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--deterministic', action='store_true', help='Use the same native deterministic controls on original and routed runs; no custom numerical kernels')
    parser.add_argument('--reference-defer-kitti', action='store_true', help='Explicit reference import adapter for PointPillar only; does not validate or replace the original KITTI evaluator')
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--variant', choices=('original', 'repeat', 'routed'))
    parser.add_argument('--purpose', choices=('training', 'inference'))
    args = parser.parse_args()
    worker(args) if args.worker else main(args)
