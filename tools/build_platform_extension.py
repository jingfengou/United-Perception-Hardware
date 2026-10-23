"""Build into a task checkout; capture the complete compiler log and exit status."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", choices=("nvidia", "metax"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    if args.log.exists() or args.log.with_suffix('.json').exists():
        raise FileExistsError('Select a new build log; keep prior evidence')
    env = os.environ.copy()
    env.update(UP_BUILD_BACKEND=args.vendor, MAX_JOBS="2")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    if args.vendor == "nvidia":
        env.setdefault("CUDA_HOME", "/usr/local/cuda")
    else:
        env.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    started = time.time()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "setup.py", "build_ext", "--inplace"]
    with args.log.open("w") as output:
        result = subprocess.run(command, cwd=args.root, env=env, stdout=output, stderr=subprocess.STDOUT)
    summary = {"vendor": args.vendor, "command": command, "returncode": result.returncode,
               "seconds": time.time() - started, "log_sha256": hashlib.sha256(args.log.read_bytes()).hexdigest()}
    args.log.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if result.returncode:
        print("\n".join(args.log.read_text(errors="replace").splitlines()[-60:]))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
