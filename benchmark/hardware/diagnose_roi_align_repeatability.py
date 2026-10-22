"""Diagnose native RoIAlign backward repeatability without replacing its kernel."""
import argparse
import collections
import collections.abc
import hashlib
import json
import os
from pathlib import Path
import sys


def run(source, repeats):
    import numpy as np
    import torch

    collections.Iterable = collections.abc.Iterable
    np.float = float
    os.environ["DEFAULT_TASKS"] = "cls:det"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, str(source.resolve()))
    import up
    initialize = vars(up).get("initialize")
    if initialize is not None and getattr(initialize, "__module__", None) == "up":
        initialize()
    from up.extensions.python.roi_align import RoIAlignFunction

    torch.cuda.set_device(0)
    torch.manual_seed(131)
    torch.use_deterministic_algorithms(True)
    features = torch.randn(2, 64, 32, 48, device="cuda", requires_grad=True)
    xy = torch.rand(256, 2, device="cuda") * 40
    wh = torch.rand(256, 2, device="cuda") * 60 + 4
    rois = torch.cat([torch.arange(256, device="cuda").remainder(2).float()[:, None], xy, xy + wh], dim=1)
    output = RoIAlignFunction.apply(features, rois, 7, 7, .25, 0, "AVG")
    upstream = torch.randn_like(output)
    gradients = [torch.autograd.grad(output, features, upstream, retain_graph=True)[0].detach().cpu()
                 for _ in range(repeats)]
    deltas = [float((gradient - gradients[0]).abs().max()) for gradient in gradients[1:]]
    kernel = source / "up/extensions/csrc/roi_align/roi_align_kernel.cu"
    return {"scope": "diagnostic_only; fixed native input and upstream gradient; not model precision acceptance",
            "diagnostic_completed": True, "source_root": str(source), "repeats": repeats,
            "deterministic_algorithms_requested": True,
            "bitwise_equal": all(torch.equal(value, gradients[0]) for value in gradients[1:]),
            "max_abs_gradient_differences": deltas,
            "max_abs_gradient_difference": max(deltas, default=0.),
            "input_shape": list(features.shape), "rois": 256, "dtype": str(features.dtype),
            "kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
            "kernel_normalized_sha256": hashlib.sha256(kernel.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(0)}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    report = run(args.source, args.repeats)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2))
