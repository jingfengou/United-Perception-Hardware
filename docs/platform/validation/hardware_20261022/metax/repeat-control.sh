#!/usr/bin/env bash
set -eu
ROOT=/workspace/up-run/platform-validation-20261022
PYTHON=/workspace/up-run/venv/bin/python
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
export PYTORCH_DISABLE_CUDA_CUDNN_TF32=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export USER=oujingfeng LOGNAME=oujingfeng
mkdir "$ROOT/repeat-01"
for model in resnet18 yolov5s detr vit_b16; do
  "$PYTHON" "$ROOT/changed/tools/validate_platform_equivalence.py" training \
    --source "$ROOT/baseline" --data "$ROOT/run-01/data/data.json" \
    --output "$ROOT/repeat-01/$model" --backend metax --model "$model" \
    > "$ROOT/repeat-01/$model.log" 2>&1
  echo "baseline repeat $model finished"
done
