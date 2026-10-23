"""Optional vendor build. Public metadata/builds never import PyTorch or probe GPUs."""
import os
from pathlib import Path

from setuptools import setup


def get_extensions():
    mode = os.environ.get("UP_BUILD_BACKEND", "none")
    if mode in ("none", "auto"):
        return []
    if mode != "nvidia":
        raise RuntimeError("UP_BUILD_BACKEND must be none, auto (no compilation) or nvidia")
    try:
        import torch
        from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME
    except ImportError as exc:
        raise RuntimeError("NVIDIA build requires the selected NVIDIA PyTorch environment") from exc
    if not torch.version.cuda or getattr(torch.version, "hip", None) or CUDA_HOME is None:
        raise RuntimeError("NVIDIA build requires NVIDIA PyTorch and CUDA_HOME/nvcc")
    if (getattr(torch.version, "maca", None) or getattr(torch.version, "metax", None)
            or any(name in torch.__version__.lower() for name in ("metax", "maca"))):
        raise RuntimeError("NVIDIA binaries cannot be built using MetaX PyTorch")
    package = Path(__file__).parent / "src/up_platform"
    root = package / "base/csrc/nvidia"
    operator_roots = sorted((package / "ops").glob("*/nvidia/csrc"))
    sources = sorted(str(path.relative_to(Path(__file__).parent)) for pattern in ("*.cpp", "*.cu")
                     for directory in [root] + operator_roots for path in directory.rglob(pattern))
    return [CUDAExtension(
        "up_platform.backends.nvidia._C", sources,
        include_dirs=[str(directory.resolve()) for directory in [root] + operator_roots],
        define_macros=[("WITH_CUDA", None)], extra_compile_args={"cxx": [], "nvcc": [
            "-DCUDA_HAS_FP16=1", "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__"]})]


if __name__ == "__main__":
    extensions = get_extensions()
    commands = {}
    if extensions:
        from torch.utils.cpp_extension import BuildExtension
        commands["build_ext"] = BuildExtension
    setup(ext_modules=extensions, cmdclass=commands)
