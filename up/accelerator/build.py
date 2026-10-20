"""Loaded by setup.py without importing the up package or requiring a GPU."""
import os
from pathlib import Path


def get_extensions():
    mode = os.environ.get("UP_BUILD_BACKEND", "auto")
    if mode not in ("auto", "nvidia", "none"):
        raise RuntimeError("UP_BUILD_BACKEND must be auto, nvidia or none")
    if mode == "none":
        return []
    try:
        import torch
        from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME
    except ImportError:
        if mode == "auto":
            return []
        raise RuntimeError("NVIDIA extension build requires PyTorch")
    selected = os.environ.get("UP_ACCELERATOR_BACKEND", "auto")
    if mode == "auto" and (selected not in ("auto", "nvidia") or not torch.version.cuda or not CUDA_HOME):
        return []
    if CUDA_HOME is None:
        raise RuntimeError("NVIDIA extension build requires CUDA_HOME/nvcc")
    source_root = Path("up") / "extensions" / "csrc"
    sources = sorted(str(p) for suffix in ("*.cpp", "*.cu") for p in source_root.rglob(suffix))
    return [CUDAExtension(
        "up.extensions._C", sources, include_dirs=[str(source_root.resolve())],
        define_macros=[("WITH_CUDA", None)],
        extra_compile_args={"cxx": [], "nvcc": [
            "-DCUDA_HAS_FP16=1", "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__"]})]
