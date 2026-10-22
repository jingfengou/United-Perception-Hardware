"""Bounded real-device runtime and portable 2D-operation checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(args):
    os.environ.setdefault("DEFAULT_TASKS", "cls:det")
    from validate_training_accuracy import prepare_legacy_environment
    prepare_legacy_environment()
    from up.accelerator import configure_accelerator, get_accelerator, ops
    configure_accelerator(backend=args.backend, reset=True)
    accelerator = get_accelerator()
    accelerator.set_device(args.device)
    if accelerator.device_type == "cpu":
        raise ValueError("This probe requires a real accelerator, not the CPU backend")
    assert accelerator.health_check()
    device = accelerator.device
    moved = accelerator.to_device({"value": [torch.ones(3)]})["value"][0]
    assert moved.device == device and accelerator.is_accelerator_tensor(moved)
    accelerator.manual_seed(20261021)
    state = accelerator.get_rng_state()
    expected = torch.rand(8, device=device)
    accelerator.set_rng_state(state)
    assert torch.equal(expected, torch.rand(8, device=device))
    assert accelerator.float_tensor(torch.Size([2, 3])).shape == (2, 3)
    assert accelerator.int_tensor([1, 2]).dtype == torch.int32
    accelerator.reset_peak_memory_stats()
    accelerator.empty_cache()
    assert float(moved.sum()) == 3
    accelerator.synchronize()
    memory = {"allocated": accelerator.memory_allocated(),
              "peak_allocated": accelerator.max_memory_allocated()}
    assert all(isinstance(value, int) and value >= 0 for value in memory.values())
    from up.flags import ALIGNED_FLAG
    original = ALIGNED_FLAG.aligned, ALIGNED_FLAG.offset
    cases = []
    try:
        for offset in (0, 1):
            ALIGNED_FLAG.aligned, ALIGNED_FLAG.offset = offset == 0, offset
            for dtype in (torch.float32, torch.float16):
                boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9],
                                      [20, 20, 30, 30]], dtype=dtype, device=device)
                scores = boxes.new_tensor([.9, .8, .7])
                keep = ops.naive_nms(torch.cat([boxes, scores[:, None]], dim=1), .5)
                assert keep.device == device and keep.dtype == torch.int64
                assert keep.tolist() == [0, 2]
                overlap = ops.gpu_iou_overlap(boxes[:1], boxes[1:2])
                # Overlap keeps inclusive corners independently of NMS alignment.
                ratio = 81 / 121
                assert overlap.device == device
                torch.testing.assert_close(overlap, boxes.new_tensor([[ratio]]),
                                           atol=1e-3, rtol=1e-3)
                assert ops.gpu_iou_overlap(boxes[:0], boxes).shape == (0,)
                cases.append({"offset": offset, "dtype": str(dtype),
                              "keep": keep.tolist(), "iou": float(overlap.item())})
        boxes = torch.tensor([[20, 20, 30, 30], [0, 0, 10, 10],
                              [1, 1, 9, 9]], dtype=torch.float32, device=device)
        keep = ops.torchvision_nms(boxes, boxes.new_tensor([.7, .9, .8]), .5)
        assert keep.device == device and keep.tolist() == [1, 0]
    finally:
        ALIGNED_FLAG.aligned, ALIGNED_FLAG.offset = original
    accelerator.synchronize()
    report = {"passed": True, "backend": accelerator.name, "device": str(device),
              "gpu": accelerator.get_device_properties().name,
              "torch": torch.__version__, "torchvision_nms_keep": keep.tolist(),
              "runtime": {"health": True, "migration": True, "rng_restore": True,
                          "legacy_constructors": True, "cache_preserves_live_tensors": True,
                          "memory_bytes": memory},
              "portable_2d_cases": cases,
              "operation_owners": {name: accelerator.capabilities()["operations"][name]["owner"]
                                   for name in ("naive_nms", "gpu_iou_overlap", "torchvision_nms")},
              "nvidia_extension_loaded": any(name in sys.modules for name in (
                  "up.extensions._C", "up_platform.backends.nvidia._C")),
              "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "small fixed cases only; not full operator, model accuracy or performance qualification"}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
