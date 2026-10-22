#!/usr/bin/env bash
set -eu
ROOT=/workspace/up-platform-validation-20261022
PYTHON=/workspace/up-cross-vendor-diagnosis-20261021/venv/bin/python
export CUDA_VISIBLE_DEVICES=3 DEFAULT_TASKS=cls:det OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$ROOT/changed/packages/up-platform/src"
"$PYTHON" "$ROOT/baseline/tools/validate_native_ops.py" --device 0 --output "$ROOT/baseline-ops.pt" > "$ROOT/baseline-ops.log" 2>&1
"$PYTHON" "$ROOT/changed/tools/validate_native_ops.py" --device 0 --output "$ROOT/changed-ops.pt" > "$ROOT/changed-ops.log" 2>&1
"$PYTHON" "$ROOT/changed/tools/validate_native_ops.py" --compare "$ROOT/baseline-ops.pt" "$ROOT/changed-ops.pt" --output "$ROOT/native-ops-comparison.json"
