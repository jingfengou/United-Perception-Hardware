"""Public installation never probes CUDA; compile only with UP_BUILD_BACKEND."""
import os
import re
from pathlib import Path
from setuptools import find_packages, setup

version = re.search(r'__version__ = "(\D*)(.*?)"', Path("up/__init__.py").read_text()).group(2)
extensions, commands = [], {}
vendor = os.environ.get("UP_BUILD_BACKEND")
if vendor:
    from up_platform.ops._native.build import extensions as build_extensions
    extensions, commands = build_extensions(vendor)

native_root = Path("up_platform/ops/_native")
native_sources = sorted(p.relative_to(native_root).as_posix() for p in (native_root / "csrc").rglob("*")
                        if p.suffix in (".h", ".cpp", ".cu"))

setup(name="UP", version=version, author="Model ToolChain", description="United Perception",
      python_requires=">=3.10", packages=find_packages(), ext_modules=extensions,
      cmdclass=commands, include_package_data=False,
      package_data={"up_platform.ops._native": native_sources})
