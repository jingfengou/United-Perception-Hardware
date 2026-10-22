"""Run with torchrun --standalone --nproc_per_node=2."""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from up.accelerator import configure_accelerator, get_accelerator, distributed
from up.accelerator.layers import SyncBatchNorm


def main():
    configure_accelerator(backend="nvidia")
    accelerator = get_accelerator()
    accelerator.set_device(int(os.environ["LOCAL_RANK"]))
    distributed.init_process_group()
    try:
        rank = distributed.get_rank()
        device = accelerator.device
        value = torch.tensor([rank + 1.], device=device)
        distributed.all_reduce(value)
        assert value.item() == 3.
        torch.manual_seed(20261020)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 1, bias=False), SyncBatchNorm(4),
            torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
            torch.nn.Linear(4, 2)).to(device)
        model = distributed.wrap_model(model, device_ids=[accelerator.current_device()],
                                       broadcast_buffers=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(100 + rank)
            loss = model(torch.randn(2, 3, 8, 8, device=device)).square().mean()
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
        parameters = torch.cat([p.detach().flatten() for p in model.parameters()])
        gathered = [torch.empty_like(parameters) for _ in range(2)]
        distributed.all_gather(gathered, parameters)
        torch.testing.assert_close(gathered[0], gathered[1], atol=1e-6, rtol=1e-6)
        from up.utils.env import dist_helper
        payload = {"source": "rank0", "items": list(range(17))} if rank == 0 else None
        received = dist_helper.broadcast_object(payload)
        assert received == {"source": "rank0", "items": list(range(17))}
        objects = dist_helper.all_gather({"rank": rank})
        assert objects == [{"rank": 0}, {"rank": 1}]
        distributed.barrier()
        if rank == 0:
            output = Path(os.environ["UP_VALIDATION_OUTPUT"])
            output.write_text(json.dumps({
                "passed": True, "world_size": 2, "all_reduce": True,
                "ddp_parameter_equality": True, "sync_batch_norm_training": True,
                "object_broadcast": True, "object_all_gather": True,
                "communication_backend": str(distributed.get_backend()),
                "torch": torch.__version__, "cuda": torch.version.cuda,
            }, indent=2))
            print("DISTRIBUTED_VALIDATION_PASSED", flush=True)
    finally:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
