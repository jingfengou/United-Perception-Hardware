"""Read-only probe; does not allocate a GPU or modify an existing environment."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(args):
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=20)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def probe():
    packages = {}
    for name in ("torch", "torchvision", "triton", "spconv", "cumm", "mqbench", "msbench", "linklink",
                 "numpy", "scipy", "onnx", "ninja", "setuptools", "pytest"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    processes = []
    for entry in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = entry.read_bytes().replace(b"\0", b" ").decode(errors="replace")
            executable = Path(argv.split(" ", 1)[0]).name.lower()
            if any(word in executable for word in ("python", "supervisor", "worker", "queue", "train")):
                processes.append({"pid": int(entry.parent.name), "uid": entry.stat().st_uid, "command": argv[:500]})
        except (OSError, ValueError):
            pass
    result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "packages": packages, "processes": processes,
              "compilers": {name: shutil.which(name) for name in ("nvcc", "mxcc", "mxc++", "clang++", "g++", "ninja")},
              "sdk_paths": {name: os.environ.get(name) for name in ("CUDA_HOME", "MACA_PATH", "MACA_HOME")}}
    try:
        import torch
        from torch.utils import cpp_extension
        result["torch"] = {"version": torch.__version__, "cuda_version": torch.version.cuda,
                           "cpp_extension": cpp_extension.__file__,
                           "extension_factories": [name for name in dir(cpp_extension) if "Extension" in name or "HOME" in name],
                           "cuda_home": str(getattr(cpp_extension, "CUDA_HOME", None))}
        source = Path(cpp_extension.__file__).read_text()
        result["compiler_integration"] = [line for line in source.splitlines()
                                           if any(token in line.lower() for token in ("mxcc", "maca", "hipify"))][:100]
    except Exception as error:
        result["torch_error"] = repr(error)
    result["gpu_inventory"] = command(["nvidia-smi", "--query-gpu=index,name,uuid,memory.used,utilization.gpu", "--format=csv,noheader"]
                                      if shutil.which("nvidia-smi") else ["mx-smi"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = probe()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
