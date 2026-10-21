"""Audit completed, locally generated training checkpoints and capture package provenance."""
import argparse
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tensors(value, path="state"):
    import torch

    count, failures = 0, []
    if isinstance(value, torch.Tensor):
        count = 1
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            failures.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            number, invalid = finite_tensors(child, path + "." + str(key))
            count += number
            failures.extend(invalid)
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            number, invalid = finite_tensors(child, path + "." + str(index))
            count += number
            failures.extend(invalid)
    elif isinstance(value, float) and not math.isfinite(value):
        failures.append(path)
    return count, failures


def audit(directory):
    import importlib.util
    import torch

    spec = importlib.util.spec_from_file_location("accuracy_driver", Path(__file__).with_name("validate_training_accuracy.py"))
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    report = json.loads((directory / "report.json").read_text())
    if report["status"] != "completed":
        raise ValueError("Only complete training runs are eligible for the checkpoint audit")
    checkpoint = directory / "comparison_state.pt"
    # These are our own training artifacts, including the original scheduler's Python state.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if set(state) != {"model", "optimizer", "scheduler"}:
        raise ValueError("Incomplete training state")
    result = {"report_sha256": sha256(directory / "report.json"), "checkpoint_sha256": sha256(checkpoint),
              "auditor_sha256": sha256(__file__), "sections": {}, "status": "passed"}
    for name, value in state.items():
        count, failures = finite_tensors(value, name)
        result["sections"][name] = {"tensor_count": count, "non_finite": failures}
        if failures:
            raise ValueError("Non-finite saved tensors: " + str(failures))
    if driver.state_hash(state["model"]) != report["final_state_sha256"]:
        raise ValueError("Saved model no longer matches its training report")
    if not result["sections"]["model"]["tensor_count"] or not result["sections"]["optimizer"]["tensor_count"]:
        raise ValueError("Empty model or optimizer tensor state")
    (directory / "checkpoint-audit.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def environment():
    import cv2
    import numpy
    import PIL
    import scipy
    import torch
    import torchvision

    packages = {}
    for distribution in metadata.distributions():
        packages.setdefault(distribution.metadata.get("Name") or "unknown", distribution.version)
    return {"python": platform.python_version(), "executable": sys.executable,
            "torch": torch.__version__, "torch_path": torch.__file__, "cuda_compatibility_version": torch.version.cuda,
            "packages": dict(sorted(packages.items())), "auditor_sha256": sha256(__file__),
            "active_imports": {name: {"version": module.__version__, "path": module.__file__}
                               for name, module in (("cv2", cv2), ("numpy", numpy), ("PIL", PIL),
                                                    ("scipy", scipy), ("torchvision", torchvision))},
            "cpu_threads": {"torch": torch.get_num_threads(), "opencv": cv2.getNumThreads()},
            "precision_readback": {"matmul": torch.get_float32_matmul_precision(),
                                   "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                                   "cudnn_tf32": torch.backends.cudnn.allow_tf32},
            "environment": {key: os.environ.get(key) for key in (
                "PYTORCH_DISABLE_CUDA_CUDNN_TF32", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "DEFAULT_TASKS", "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+")
    parser.add_argument("--environment", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.environment:
        result = environment()
    else:
        if not args.runs:
            parser.error("--runs is required for checkpoint audits")
        result = {str(path): audit(path) for path in args.runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({"output": str(args.output), "status": "completed"}, sort_keys=True))


if __name__ == "__main__":
    main()
