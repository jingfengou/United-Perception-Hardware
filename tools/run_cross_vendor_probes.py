"""Run isolated numerical controls sequentially on one explicitly idle GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


CASES = {
    "rng-wide": ["--kind", "rng", "--rng-wide"],
    "ops": ["--kind", "ops"],
    "vit-paired": ["--kind", "training", "--model", "vit_b16", "--steps", "2", "--paired-masks"],
    "detr-native": ["--kind", "training", "--model", "detr", "--steps", "4"],
    "detr-paired": ["--kind", "training", "--model", "detr", "--steps", "2", "--paired-masks"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("nvidia", "metax"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    args = parser.parse_args()
    if args.device < 0 or (args.output.exists() and any(args.output.iterdir())):
        parser.error("Need a nonnegative physical device and a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    driver = Path(__file__).with_name("probe_cross_vendor_numerics.py")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device), PYTHONHASHSEED="0",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", DEFAULT_TASKS="cls:det",
                       PYTORCH_DISABLE_CUDA_CUDNN_TF32="1", TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0")
    manifest = {"scope": "diagnostic_only", "backend": args.backend, "device": args.device,
                "probe_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(), "cases": {}}
    for name in args.cases:
        command = [sys.executable, str(driver), "--backend", args.backend,
                   "--source", str(args.source.resolve()), "--data", str(args.data.resolve()),
                   "--output", str((args.output / name).resolve()), *CASES[name]]
        with (args.output / (name + ".log")).open("w") as log:
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        manifest["cases"][name] = {"command": command, "exit_code": result.returncode}
        (args.output / "execution.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        if result.returncode:
            raise RuntimeError("Probe failed; all partial evidence preserved: " + name)
        print(json.dumps({"completed_probe": name}), flush=True)
    manifest["status"] = "completed"
    (args.output / "execution.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
