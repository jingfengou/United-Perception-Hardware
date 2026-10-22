#!/usr/bin/env bash
set -eu
ROOT=$1
PYTHON=$2
BACKEND=$3
DATA=$4
DEVICE=$5
export CUDA_VISIBLE_DEVICES="$DEVICE" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
export PYTORCH_DISABLE_CUDA_CUDNN_TF32=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export USER=${USER:-oujingfeng} LOGNAME=${LOGNAME:-oujingfeng}
"$PYTHON" "$ROOT/changed/tools/validate_platform_equivalence.py" suite \
  --baseline "$ROOT/baseline" --changed "$ROOT/changed" --data "$DATA" \
  --backend "$BACKEND" --output "$ROOT/run-01"
