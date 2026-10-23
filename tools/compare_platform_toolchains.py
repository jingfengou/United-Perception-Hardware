"""Execute real optional packages, comparing direct calls with platform bindings.

Each case/route runs in a fresh process because ASP owns process-global state.
No skipped import, no-op scheduler, or ordinary optimizer counts as SDK validation.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import torch

from compare_platform_models import compare, cpu


def worker(args):
    from up_platform import configure_platform
    p = configure_platform({'backend': args.vendor}, purpose='training')
    p.device.set_device(0)
    torch.manual_seed(131)
    np.random.seed(131)
    p.device.manual_seed(131)
    device = p.device.device()
    routed = args.route == 'routed'
    record = {}
    if args.case == 'sparse':
        from up_platform.ops.sparse.torch_impl import spconv
        package = spconv()
        ops = p.category('sparse') if routed else package
        def build_sparse():
            return ops.SparseSequential(
                ops.SubMConv3d(4, 8, 3, padding=1, indice_key='s'), torch.nn.ReLU(),
                ops.SparseConv3d(8, 8, 3, stride=2, padding=1, indice_key='d'),
                ops.SparseInverseConv3d(8, 4, 3, indice_key='d'),
            ).to(device)
        model = build_sparse()
        indices = torch.tensor([[0, z, y, x] for z in range(2, 6) for y in range(2, 6) for x in range(2, 6)], dtype=torch.int32, device=device)
        features = torch.randn(len(indices), 4, device=device, requires_grad=True)
        tensor = ops.SparseConvTensor(features, indices, [8, 8, 8], 1)
        output = model(tensor)
        def dense_value(value):
            if routed:
                return p.category('sparse').dense(value)
            result = value.dense()
            # Explicit reference adapter: MetaX returns N...C, UP requires NC...
            if args.vendor == 'metax':
                result = result.permute(0, result.ndim-1, *range(1, result.ndim-1)).contiguous()
            return result
        dense = dense_value(output)
        dense.square().mean().backward()
        record = {'output': cpu(dense), 'indices': cpu(output.indices), 'input_grad': cpu(features.grad),
                  'gradients': {n: cpu(v.grad) for n, v in model.named_parameters()}, 'state': cpu(model.state_dict())}
        fresh = build_sparse()
        fresh.load_state_dict(record['state'], strict=True)
        record['restored'] = cpu(dense_value(fresh(ops.SparseConvTensor(features.detach().clone(), indices, [8, 8, 8], 1))))
        torch.testing.assert_close(record['output'], record['restored'], rtol=1e-4, atol=1e-5)
    elif args.case == 'sparse_block':
        # This exercises the lazy UP SparseModule inheritance, not just spconv.
        from up.tasks.det_3d.models.backbones_3d.spconv_backbone import SparseBasicBlock
        from up_platform.ops.sparse.torch_impl import spconv
        ops = p.category('sparse')
        block = SparseBasicBlock(8, 8, norm_fn=torch.nn.BatchNorm1d, indice_key='b').to(device)
        assert isinstance(block, spconv().SparseModule)
        indices = torch.tensor([[0, z, y, x] for z in range(2, 5) for y in range(2, 5) for x in range(2, 5)], dtype=torch.int32, device=device)
        features = torch.randn(len(indices), 8, device=device, requires_grad=True)
        output = ops.SparseSequential(block)(ops.SparseConvTensor(features, indices, [8, 8, 8], 1))
        output.features.square().mean().backward()
        record = {'output': cpu(output.features), 'gradients': cpu(features.grad), 'state': cpu(block.state_dict())}
    elif args.case == 'qdrop':
        from easydict import EasyDict
        from mqbench.prepare_by_platform import prepare_by_platform, BackendType
        from mqbench.utils.state import enable_calibration_woquantization
        from mqbench.advanced_ptq import ptq_reconstruction
        ops = p.category('quantization')
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
                                    torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(8, 4)).eval().to(device)
        config = {'extra_qconfig_dict': {'w_observer': 'MSEObserver', 'a_observer': 'EMAMSEObserver',
                   'a_fakequantize': 'QDropFakeQuantize', 'w_fakequantize': 'AdaRoundFakeQuantize',
                   'w_qscheme': {'bit': 2, 'symmetry': False, 'per_channel': True, 'pot_scale': False, 'p': 2.4},
                   'a_qscheme': {'bit': 4, 'symmetry': False, 'per_channel': False, 'pot_scale': False, 'p': 2.4}}}
        prepare = ops.prepare if routed else prepare_by_platform
        model = prepare(model, BackendType.Academic, config).to(device)
        calibrate = ops.enable_calibration if routed else enable_calibration_woquantization
        calibrate(model)
        data = [torch.randn(2, 3, 8, 8, device=device) for _ in range(2)]
        for batch in data:
            model(batch)
        config = EasyDict(pattern='block', scale_lr=4e-5, warm_up=.25, weight=.005,
                          max_count=4, b_range=[20, 2], keep_gpu=True, round_mode='learned_hard_sigmoid', prob=.5)
        reconstruct = ops.reconstruct if routed else ptq_reconstruction
        result = reconstruct(model, data, config)
        record = {'output': cpu(result(data[0])), 'state': cpu(result.state_dict())}
        restored = copy.deepcopy(result)
        restored.load_state_dict(record['state'], strict=True)
        record['restored'] = cpu(restored(data[0]))
        torch.testing.assert_close(record['output'], record['restored'], rtol=1e-4, atol=1e-5)
    elif args.case in ('quantization', 'quantization_export'):
        from mqbench.prepare_by_platform import prepare_by_platform, BackendType
        from mqbench.utils.state import enable_calibration_woquantization, enable_quantization
        ops = p.category('quantization')
        prepare = ops.prepare if routed else prepare_by_platform
        calibrate = ops.enable_calibration if routed else enable_calibration_woquantization
        quantize = ops.enable_quantization if routed else enable_quantization
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.BatchNorm2d(8), torch.nn.ReLU(),
                                    torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(8, 4)).to(device)
        model = prepare(model, BackendType.Tensorrt, {}).to(device)
        inputs = torch.randn(2, 3, 8, 8, device=device)
        calibrate(model)
        model(inputs)
        quantize(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        output = model(inputs)
        output.square().mean().backward()
        record['gradients'] = {n: cpu(v.grad) for n, v in model.named_parameters()}
        optimizer.step()
        record['state'] = cpu(model.state_dict())
        restored = copy.deepcopy(model)
        restored.load_state_dict(record['state'], strict=True)
        model.eval()
        restored.eval()
        record['output'] = cpu(model(inputs))
        record['restored'] = cpu(restored(inputs))
        torch.testing.assert_close(record['output'], record['restored'], rtol=1e-4, atol=1e-5)
        if args.case == 'quantization_export':
            from mqbench.convert_deploy import convert_deploy
            convert = ops.convert_deploy if routed else convert_deploy
            convert(model, BackendType.Tensorrt, dummy_input=(inputs,), model_name=args.output.stem,
                    output_path=str(args.output.parent))
            import onnx
            paths = list(args.output.parent.glob(args.output.stem+'*.onnx'))
            assert paths, 'No actual ONNX graph was written'
            record['onnx_node_types'] = [[n.op_type for n in onnx.load(path).graph.node] for path in sorted(paths)]
    elif args.case == 'amba':
        from msbench.scheduler import build_sparse_scheduler
        ops = p.category('pruning')
        directory = args.output.parent/args.output.stem
        config = {'mask_generator': {'type': 'NormalMaskGenerator'}, 'fake_sparse': {'type': 'FakeSparse'},
                  'scheduler': {'type': 'AmbaLevelPruneScheduler', 'kwargs': {'total_iters': 4, 'sparsity_table': [30],
                     'prun_algo': 1, 'prun_algo_tuning': .5, 'do_sparse_analysis': False,
                     'output_dir': str(directory/'analysis'), 'save_dir': str(directory/'checkpoints'),
                     'default_input_shape': [1, 3, 8, 8]}}}
        scheduler = ops.build_scheduler(config) if routed else build_sparse_scheduler(config)
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
                                    torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(8, 4)).to(device)
        model = ops.prepare(scheduler, model) if routed else scheduler.prepare_sparse_model(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        lr = torch.optim.lr_scheduler.StepLR(optimizer, 2)
        ops.set_lr_scheduler(scheduler, lr) if routed else scheduler.set_lr_scheduler(lr)
        ops.prune(scheduler, model, 0) if routed else scheduler.prune_model(model, 0)
        rate = ops.sparse_rate(scheduler, model) if routed else scheduler.compute_sparse_rate(model)
        assert 0 < rate < 1, 'A real nontrivial pruning mask must be produced'
        inputs = torch.randn(2, 3, 8, 8, device=device)
        output = model(inputs)
        output.square().mean().backward()
        optimizer.step()
        record = {'output': cpu(output), 'state': cpu(model.state_dict()), 'optimizer': cpu(optimizer.state_dict()),
                  'gradients': {n: cpu(v.grad) for n, v in model.named_parameters()}, 'sparse_rate': rate}
        ops.set_iteration(scheduler, 3) if routed else setattr(scheduler, 'iteration', 3)
        record['iteration'] = scheduler.iteration
        record['last_prune_step'] = ops.last_prune_step(scheduler) if routed else scheduler.get_last_prune_step()
        restored = copy.deepcopy(model)
        restored.load_state_dict(record['state'], strict=True)
        torch.testing.assert_close(model(inputs), restored(inputs), rtol=1e-4, atol=1e-5)
        ops.export(scheduler, model) if routed else scheduler.export_sparse_model(model)
        record['exported_state'] = cpu(model.state_dict())
    elif args.case == 'pruning':
        from msbench.scheduler import build_sparse_scheduler
        ops = p.category('pruning')
        config = {'scheduler': {'type': 'AmpereScheduler', 'kwargs': {'allow_permutation': False, 'verbosity': 0}}}
        scheduler = ops.build_scheduler(copy.deepcopy(config)) if routed else build_sparse_scheduler(copy.deepcopy(config))
        model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8)).to(device)
        model = ops.prepare(scheduler, model) if routed else scheduler.prepare_sparse_model(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        ops.init_optimizer(scheduler, optimizer) if routed else scheduler.init_optimizer(optimizer)
        inputs = torch.randn(4, 32, device=device)
        output = model(inputs)
        output.square().mean().backward()
        optimizer.step()
        record = {'output': cpu(output), 'state': cpu(model.state_dict()), 'optimizer': cpu(optimizer.state_dict()),
                  'gradients': {n: cpu(v.grad) for n, v in model.named_parameters()}}
        masks = [v for n, v in model.named_buffers() if 'mask' in n]
        assert masks and all(int((v == 0).sum())*2 == v.numel() for v in masks), 'Actual 2:4 masks must be present'
        for n, v in model.named_parameters():
            if n.endswith('weight'):
                assert int((v == 0).sum())*2 >= v.numel(), 'Optimizer did not preserve pruning'
        saved = copy.deepcopy(model.state_dict())
        expected = cpu(model(inputs))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1)
        model.load_state_dict(saved, strict=True)
        torch.testing.assert_close(expected, cpu(model(inputs)), rtol=1e-4, atol=1e-5)
        record['restored'] = cpu(model(inputs))
    else:
        raise ValueError(args.case)
    torch.save(record, args.output.with_suffix('.pt'))
    args.output.write_text(json.dumps({'case': args.case, 'route': args.route, 'executed': True,
                                      'reference_adaptations': ['MetaX spconv N...C to upstream NC... dense layout'] if args.case == 'sparse' and args.vendor == 'metax' else [],
                                      'environment': p.device.environment()}, indent=2))


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for case in args.cases:
        paths = {}
        for route in ('direct', 'routed'):
            path = args.output / f'{case}-{route}.json'
            if path.exists() or path.with_suffix('.log').exists():
                raise FileExistsError(path)
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--vendor', args.vendor,
                       '--case', case, '--route', route, '--output', str(path)]
            with path.with_suffix('.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            paths[route] = path
            print(case, route, result.returncode, flush=True)
        record = {'case': case}
        if all(path.exists() for path in paths.values()):
            record.update(compare(*(torch.load(p.with_suffix('.pt'), weights_only=False) for p in paths.values())))
        else:
            record.update(passed=False, execution_failures=[k for k, p in paths.items() if not p.exists()])
        records.append(record)
        (args.output/'summary.json').write_text(json.dumps({'records': records, 'rtol': 1e-4, 'atol': 1e-5}, indent=2))
    raise SystemExit(0 if all(r['passed'] for r in records) else 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendor', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['sparse', 'sparse_block', 'quantization', 'quantization_export', 'pruning'])
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--case')
    parser.add_argument('--route', choices=('direct', 'routed'))
    args = parser.parse_args()
    worker(args) if args.worker else main(args)
