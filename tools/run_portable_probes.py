"""Fixed short controls for a declared hardware profile, never accuracy runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--backend", choices=("nvidia", "metax"), required=True)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", type=int, required=True)
args = parser.parse_args()
if args.output.exists() or args.device < 0:
    parser.error("Need a new output directory and a nonnegative device")
args.output.mkdir(parents=True)
environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   PYTHONHASHSEED="0", PYTORCH_DISABLE_CUDA_CUDNN_TF32="1", TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
                   TRITON_CACHE_DIR=str(args.output.parent / "triton-cache"))
for model in ("yolov5s", "detr", "vit_b16"):
    command = [sys.executable, str(Path(__file__).with_name("probe_cross_vendor_numerics.py")),
               "--source", str(args.source), "--backend", args.backend, "--data", str(args.data),
               "--kind", "training", "--model", model, "--steps", "4", "--seed", "131",
               "--numerical-profile", "portable_v1", "--output", str(args.output / model)]
    with (args.output / (model + ".log")).open("w") as log:
        status = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if status.returncode:
        raise RuntimeError("Probe failed; evidence retained: " + model)
    print(json.dumps({"completed_probe": model}), flush=True)
