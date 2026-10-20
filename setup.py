import importlib.util
import io
import re
from setuptools import setup, find_packages

with io.open("up/__init__.py", "rt", encoding="utf8") as f:
    __version__ = re.search(r'__version__ = "(\D*)(.*?)"', f.read(), re.M).group(2)


spec = importlib.util.spec_from_file_location("up_accelerator_build", "up/accelerator/build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)
extensions = build.get_extensions()
commands = {}
if extensions:
    from torch.utils.cpp_extension import BuildExtension
    commands["build_ext"] = BuildExtension


setup(
    name="UP",
    version=__version__,
    author="Model ToolChain",
    description="United Perception",
    author_email="",
    python_requires='>=3.6',
    packages=find_packages(),
    classifiers=(
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux"),
    ext_modules=extensions,
    cmdclass=commands,
)
