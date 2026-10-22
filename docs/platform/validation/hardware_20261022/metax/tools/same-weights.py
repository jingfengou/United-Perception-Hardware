import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import torch

root = Path(sys.argv[1])
driver = root / "tools/driver-v3.py"
spec = importlib.util.spec_from_file_location("driver", driver)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
torch.set_num_threads(2)
output = root / "same-weights"
output.mkdir(exist_ok=False)
summary = json.loads((root / "run-03/summary.json").read_text())
assert tool.source_manifest(root / "changed") == summary["sources"]["changed"]
result = {"scope": "same old checkpoint and input; split inference purpose; no training", "models": {}}
for model in tool.MODELS:
    target = output / model
    training = root / "run-01/baseline" / model / "training"
    command = [sys.executable, str(driver), "inference", "--source", str(root / "changed"),
               "--data", str(root / "run-01/data/data.json"), "--output", str(target), "--backend", "metax",
               "--model", model, "--training", str(training)]
    with (output / (model + ".log")).open("w") as log:
        code = subprocess.run(command, cwd=root / "changed", stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        result["models"][model] = {"passed": False, "returncode": code}
    else:
        before = root / "run-03/baseline" / model / "inference"
        cases = {}
        for name in ("batch-predictions.pt", "image-predictions.pt"):
            a, b = [torch.load(path / name, map_location="cpu", weights_only=False) for path in (before, target)]
            cases[name] = tool.close_trees(a, b)
        result["models"][model] = {"passed": True, "comparisons": cases,
                                    "checkpoint_sha256": tool.file_hash(training / "checkpoint.pt")}
    result["passed"] = all(value["passed"] for value in result["models"].values())
    tool.write_json(output / "summary.json", result)
    print("same weights", model, result["models"][model]["passed"], flush=True)
sys.exit(not result["passed"])
