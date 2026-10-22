"""One fixed seed, complete existing small-data budgets, two frozen revisions.

This entry point never resumes the paused multi-seed or cross-vendor cohorts.
It only orchestrates the existing real Runner driver and its strict comparator.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_cross_vendor_data import verify_data
from validate_platform_equivalence import file_hash, source_manifest, write_json


SEED = 131
BUDGETS = {"resnet18": (20, "multiseed_v1", 3140),
           "yolov5s": (300, "multiseed_v2", 600),
           "detr": (12, "multiseed_v2", 720),
           "vit_b16": (20, "multiseed_v2", 3140)}


def run(args):
    if args.backend != "metax" and args.numerical_profile != "native":
        raise ValueError("The MetaX numerical profile cannot be used on another vendor")
    if not args.device.isdecimal():
        raise ValueError("Pass one actually allocated physical device")
    args.output.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.data.read_text())
    integrity = verify_data(data)
    epochs, protocol, expected_steps = BUDGETS[args.model]
    driver = Path(__file__).with_name("validate_training_accuracy.py")
    sources = {key: source_manifest(value) for key, value in (
        ("baseline", args.baseline), ("changed", args.changed))}
    manifest = {"scope": "single-seed complete CIFAR-10/PennFudan regression; not full ImageNet/COCO or statistical equivalence",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "model": args.model, "backend": args.backend, "seed": SEED,
                "epochs": epochs, "expected_steps_per_version": expected_steps,
                "numerical_profile": args.numerical_profile, "physical_device": args.device,
                "data_sha256": file_hash(args.data), "verified_data": integrity,
                "sources": sources, "driver_sha256": file_hash(driver),
                "controller_sha256": file_hash(__file__),
                "tolerances": {"rtol": 1e-4, "atol": 1e-5, "metric_atol": 1e-6},
                "selection": "final epoch only; no best checkpoint or seed selection", "jobs": []}
    write_json(args.output / "execution.json", manifest)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.device, OMP_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", PYTHONHASHSEED="0", DEFAULT_TASKS="cls:det",
                       CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTORCH_DISABLE_CUDA_CUDNN_TF32="1",
                       TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0")
    for label, source in (("baseline", args.baseline), ("changed", args.changed)):
        command = [sys.executable, str(driver), "--source", str(source), "--data", str(args.data),
                   "--model", args.model, "--backend", args.backend, "--protocol", protocol,
                   "--epochs", str(epochs), "--seed", str(SEED), "--audit-batches",
                   "--numerical-profile", args.numerical_profile, "--output", str(args.output / label)]
        with (args.output / (label + ".log")).open("w") as log:
            result = subprocess.run(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
        job = {"label": label, "command": command, "exit_code": result.returncode}
        manifest["jobs"].append(job)
        write_json(args.output / "execution.json", manifest)
        print(args.model, label, "PASS" if result.returncode == 0 else "FAIL", flush=True)
        if result.returncode:
            raise RuntimeError("Training failed; retained original outputs: " + str(args.output / label))
        report = json.loads((args.output / label / "report.json").read_text())
        if report["status"] != "completed" or report["training_steps"] != expected_steps:
            raise ValueError("The complete predeclared training budget was not executed")
        if report["environment"]["numerical_profile"] != args.numerical_profile:
            raise ValueError("Requested numerical profile was not actually active")
        if sources[label] != source_manifest(source) or file_hash(args.data) != manifest["data_sha256"]:
            raise ValueError("Source or data manifest changed during training")
    command = [sys.executable, str(driver), "--compare", str(args.output / "baseline"),
               str(args.output / "changed"), "--output", str(args.output / "comparison.json")]
    with (args.output / "comparison.log").open("w") as log:
        result = subprocess.run(command, env=dict(environment, CUDA_VISIBLE_DEVICES=""),
                                stdout=log, stderr=subprocess.STDOUT)
    manifest.update(comparison_exit_code=result.returncode, completed=True,
                    passed=result.returncode == 0, completed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output / "execution.json", manifest)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "changed", "data", "output"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--backend", choices=("nvidia", "metax"), required=True)
    parser.add_argument("--model", choices=tuple(BUDGETS), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--numerical-profile", choices=("native", "metax_deterministic_v1"), default="native")
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
