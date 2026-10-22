"""Preserve initial evidence and complete only its failed inference phases."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

root, backend = Path(sys.argv[1]), sys.argv[2]
driver = root / "tools/driver-v3.py"
spec = importlib.util.spec_from_file_location("driver", driver)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
old = root / "run-01"
output = root / "run-03"
output.mkdir(exist_ok=False)
(output / "data").symlink_to(old / "data", target_is_directory=True)
first = json.loads((old / "summary.json").read_text())
result = {"scope": first["scope"], "backend": backend, "seed": tool.SEED, "steps": tool.STEPS,
          "rtol": tool.RTOL, "atol": tool.ATOL, "driver_sha256": tool.file_hash(driver),
          "sources": first["sources"], "reused_training": {}, "reused_inference": {}, "models": {}, "failures": {}}
for label in ("baseline", "changed"):
    assert tool.source_manifest(root / label) == first["sources"][label], "Runtime source changed"
for model in tool.MODELS:
    for label in ("baseline", "changed"):
        completed = old / label / model / "training"
        report = json.loads((completed / "report.json").read_text())
        assert report["passed"], "Cannot reuse a failed training run"
        parent = output / label / model
        parent.mkdir(parents=True)
        (parent / "training").symlink_to(completed, target_is_directory=True)
        result["reused_training"][label + "/" + model] = {
            "path": str(completed), "report_sha256": tool.file_hash(completed / "report.json"),
            "checkpoint_sha256": tool.file_hash(completed / "checkpoint.pt"),
            "driver_sha256": report["driver_sha256"]}
        prior = root / "run-02" / label / model / "inference"
        if not (prior / "report.json").is_file() or not json.loads((prior / "report.json").read_text())["passed"]:
            prior = old / label / model / "inference"
        prior_report = json.loads((prior / "report.json").read_text())
        if prior_report["passed"]:
            (parent / "inference").symlink_to(prior, target_is_directory=True)
            result["reused_inference"][label + "/" + model] = {
                "path": str(prior), "report_sha256": tool.file_hash(prior / "report.json")}
            code = 0
        else:
            command = [sys.executable, str(driver), "inference", "--source", str(root / label),
                       "--data", str(old / "data/data.json"), "--output", str(parent / "inference"),
                       "--backend", backend, "--model", model, "--training", str(completed)]
            with (parent / "inference.log").open("w") as log:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=root / label).returncode
        print(label, model, "inference", code, flush=True)
        if code:
            result["failures"][label + "/" + model + "/inference"] = code
    try:
        result["models"][model] = tool.compare_model(output, model)
        if not result["models"][model]["passed"]:
            result["failures"][model + "/comparison"] = result["models"][model]["failures"]
    except Exception as exc:
        result["failures"][model + "/comparison"] = repr(exc)
    result["passed"] = not result["failures"]
    tool.write_json(output / "summary.json", result)
sys.exit(not result["passed"])
