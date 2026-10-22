import io
import re
from setuptools import setup, find_packages

with io.open("up/__init__.py", "rt", encoding="utf8") as f:
    __version__ = re.search(r'__version__ = "(\D*)(.*?)"', f.read(), re.M).group(2)


setup(
    name="UP",
    version=__version__,
    author="Model ToolChain",
    description="United Perception",
    author_email="",
    python_requires='>=3.8',
    packages=find_packages(include=["up", "up.*"]),
    install_requires=["up-platform>=0.1.0,<0.2"],
    classifiers=(
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux"),
)
