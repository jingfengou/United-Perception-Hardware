"""Explicit native build. Never execute during public-package import."""
from pathlib import Path


def extensions(vendor):
    if vendor not in ("nvidia", "metax"):
        raise ValueError("UP_BUILD_BACKEND must be nvidia or metax")
    import torch
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension

    metax = bool(getattr(torch._utils_internal, "USE_MACA", False)) or "metax" in torch.__version__
    if metax != (vendor == "metax"):
        raise RuntimeError(f"Refusing {vendor} extension build with torch {torch.__version__}")
    root = Path(__file__).resolve().parent / "csrc"
    sources = sorted(str(p) for p in root.rglob("*") if p.suffix in (".cpp", ".cu"))
    extension = CUDAExtension(
        f"up_platform.backends._{vendor}_ops", sources,
        include_dirs=[str(root)], define_macros=[("WITH_CUDA", None)],
        extra_compile_args={"cxx": [], "nvcc": ["-DCUDA_HAS_FP16=1", "-D__CUDA_NO_HALF_OPERATORS__",
                                                  "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__"]})
    return [extension], {"build_ext": BuildExtension}
