"""Two real devices: process groups, async reductions, DDP, SyncBN, AMP, restore.

Run with torchrun --standalone --nproc-per-node=2 after checking device ownership.
This validates the selected Torch communication path, never spring.linklink.
"""
import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from up_platform import configure_platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendor', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    p = configure_platform({'backend': args.vendor, 'device': f'cuda:{rank}'}, purpose='training')
    p.device.set_device(rank)
    ops = p.category('communication')
    ops.init_process_group('nccl')
    try:
        value = torch.tensor([rank+1.], device=p.device.device())
        work = ops.all_reduce(value, async_op=True)
        work.wait()
        assert value.item() == 3
        objects = [None, None]
        ops.all_gather_object(objects, {'rank': rank})
        assert objects == [{'rank': 0}, {'rank': 1}]
        obj = [{'value': 42} if rank == 0 else None]
        ops.broadcast_object_list(obj)
        assert obj[0] == {'value': 42}
        group = ops.new_group([0, 1])
        assert ops.get_world_size(group) == 2
        ops.barrier(group)
        torch.manual_seed(131)
        base = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.SyncBatchNorm(4),
                                   torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 2))
        outputs = []
        for routed in (False, True):
            model = copy.deepcopy(base).to(p.device.device())
            model = p.execution.wrap_ddp(model, device_ids=[rank]) if routed else torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
            optimizer = p.execution.optimizer('SGD', params=model.parameters(), lr=.01, momentum=.9) if routed else torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
            scaler = p.execution.grad_scaler() if routed else torch.amp.GradScaler('cuda')
            torch.manual_seed(500+rank)
            data = torch.randn(2, 3, 8, 8, device=p.device.device())
            for _ in range(2):
                optimizer.zero_grad()
                autocast = p.execution.autocast if routed else lambda: torch.amp.autocast('cuda')
                with autocast():
                    loss = model(data).square().mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            state = copy.deepcopy(model.state_dict())
            optimizer_state = copy.deepcopy(optimizer.state_dict())
            scaler_state = copy.deepcopy(scaler.state_dict())
            with p.execution.validation(model):
                eval_output = model(data)
            assert model.training and p.purpose == 'training'
            assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
            assert scaler.state_dict() == scaler_state
            model.load_state_dict(state, strict=True)
            optimizer.load_state_dict(optimizer_state)
            scaler.load_state_dict(scaler_state)
            packed = torch.cat([v.detach().flatten() for v in model.parameters()])
            replicas = [torch.empty_like(packed) for _ in range(2)]
            ops.all_gather(replicas, packed)
            assert torch.equal(replicas[0], replicas[1])
            outputs.append((state, eval_output.detach().clone(), scaler_state))
        for key in outputs[0][0]:
            torch.testing.assert_close(outputs[0][0][key], outputs[1][0][key], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=1e-4, atol=1e-5)
        assert outputs[0][2] == outputs[1][2]
        result = {'rank': rank, 'passed': True, 'vendor': args.vendor, 'communication_library': ops.get_backend(),
                  'cases': ['async_all_reduce', 'object_gather', 'object_broadcast', 'subgroup', 'DDP', 'SyncBN',
                            'AMP_GradScaler', 'training_eval_no_update', 'state_restore', 'direct_routed_comparison'],
                  'rtol': 1e-4, 'atol': 1e-5, 'environment': p.device.environment()}
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output/f'rank{rank}.json'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        ops.destroy_process_group()


if __name__ == '__main__':
    main()
