"""Resume this recorded schedule without interrupting in-flight training children."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/workspace/up-hardware-routing-20261021-multiseed/five-models-v2")
sys.path.insert(0, str(ROOT / "tools"))
import run_multiseed_accuracy as suite

SOURCE = Path("/workspace/up-hardware-routing-20261020")
RESULTS = ROOT / "results"
PLAN = ROOT / "tools/extension-plan.json"
DATA = ROOT / "data.json"
DRIVER = ROOT / "tools/validate_training_accuracy.py"
SPEC = suite.load_plan(PLAN)["models"]["yolov5s"]
STATE = json.loads((ROOT / "orchestration-resume.json").read_text())


def active(pid, expected):
    path = Path("/proc") / str(pid) / "cmdline"
    try:
        return path.read_bytes().decode().split("\0")[:-1] == expected
    except FileNotFoundError:
        return False


def verify(target, seed, device):
    report = json.loads((target / "report.json").read_text())
    suite.verify_report(report, SPEC, seed, suite.sha256(DATA), suite.sha256(DRIVER),
                        "multiseed_v2", "yolov5s")
    if report["environment"]["visible_devices"] != device:
        raise ValueError("Paired physical device differs")
    return report


def compare(seed):
    targets = [RESULTS / suite.job_name("yolov5s", seed, role) for role in ("baseline", "routed")]
    target = RESULTS / ("yolov5s-seed{}-repeat0-comparison.json".format(seed))
    command = [sys.executable, str(DRIVER), "--compare", *map(str, targets), "--output", str(target)]
    for key, value in suite.load_plan(PLAN)["strict_numeric_tolerances"].items():
        command += ["--" + key.replace("_", "-"), str(value)]
    with target.with_suffix(".log").open("w") as log:
        result = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), stdout=log,
                                stderr=subprocess.STDOUT)
    if result.returncode not in (0, 1) or not json.loads(target.read_text())["preconditions_equal"]:
        raise RuntimeError("Invalid paired comparison")


def run_pair(seed, device):
    roles = ("baseline", "routed") if seed % 2 else ("routed", "baseline")
    for role in roles:
        target = RESULTS / suite.job_name("yolov5s", seed, role)
        if (target / "report.json").exists():
            verify(target, seed, device)
            continue
        if target.exists() and any(target.iterdir()):
            raise ValueError("Unexpected partial run: " + str(target))
        suite.ensure_device_idle(device)
        command = [sys.executable, str(DRIVER), "--source", str(SOURCE / ("changed" if role == "routed" else "baseline")),
                   "--data", str(DATA), "--model", "yolov5s", "--seed", str(seed), "--epochs", "300",
                   "--protocol", "multiseed_v2", "--output", str(target)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, PYTHONHASHSEED="0", OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", PYTHONPATH="/workspace/up-hardware-routing-20261021-accuracy/compat")
        suite.emit({"event": "rescheduled_start", "seed": seed, "role": role, "device": device,
                    "command": command})
        with target.with_suffix(".log").open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("Training failed: " + str(target))
        report = verify(target, seed, device)
        suite.emit({"event": "completed", "job": target.name, "metrics": report["final_metrics"],
                    "steps": report["training_steps"], "seconds": report["elapsed_seconds"]})
    compare(seed)


def finish_inflight(seed, device, next_seed):
    inflight = next(row for row in STATE["inflight_children"] if row["seed"] == seed)
    while active(inflight["pid"], inflight["argv"]):
        time.sleep(5)
    run_pair(seed, device)
    run_pair(next_seed, device)


def wait_for_vit():
    marker = RESULTS / "vit_b16-seed135-repeat0-comparison.json"
    while not marker.exists():
        time.sleep(5)
    if not json.loads(marker.read_text())["preconditions_equal"]:
        raise RuntimeError("ViT pair failed its preconditions")
    run_pair(135, "0")


def main():
    if suite.sha256(DRIVER) != STATE["driver_sha256"] or suite.sha256(DATA) != STATE["data_sha256"]:
        raise ValueError("Driver or data changed during rescheduling")
    manifest = dict(STATE, resume_script_sha256=suite.sha256(__file__))
    (RESULTS / "execution-yolov5s-resumed.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(finish_inflight, 131, "1", 133),
                   executor.submit(finish_inflight, 132, "2", 134), executor.submit(wait_for_vit)]
        for future in futures:
            future.result()
    suite.emit({"event": "selected_models_completed", "models": ["yolov5s"],
                "all_five_seeds": True, "training_children_interrupted": False})


if __name__ == "__main__":
    main()
