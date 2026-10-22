#!/usr/bin/env bash
set -eu
ROOT=/workspace/up-platform-validation-20261022
PYTHON=/workspace/up-cross-vendor-diagnosis-20261021/venv/bin/python
export CUDA_VISIBLE_DEVICES=3 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 UP_BUILD_BACKEND=nvidia
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$ROOT/baseline"
"$PYTHON" setup.py build_ext --inplace > "$ROOT/baseline-build.log" 2>&1
cd "$ROOT/changed/packages/up-platform"
"$PYTHON" setup.py build_ext --inplace > "$ROOT/changed-build.log" 2>&1
echo 'Both NVIDIA native extensions compiled successfully'
