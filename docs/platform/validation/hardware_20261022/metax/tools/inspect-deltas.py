import importlib.util
import json
from pathlib import Path
import sys
import torch

torch.set_num_threads(2)
root = Path(sys.argv[1])
result = {}
for model in ("resnet18", "yolov5s", "detr", "vit_b16"):
    a = root / "baseline" / model / "training"
    b = root / "changed" / model / "training"
    if not (b / "report.json").exists():
        continue
    ra, rb = [json.loads((p / "report.json").read_text()) for p in (a, b)]
    if not ra["passed"] or not rb["passed"]:
        result[model] = {"training_passed": False}
        continue
    entry = {"initial_exact": ra["initial_state_sha256"] == rb["initial_state_sha256"],
             "final_exact": ra["final_state_sha256"] == rb["final_state_sha256"],
             "loss_deltas": [y["loss"] - x["loss"] for x, y in zip(ra["steps"], rb["steps"])],
             "batches_exact": [x["batch_sha256"] == y["batch_sha256"] for x, y in zip(ra["steps"], rb["steps"])],
             "rng_exact": [x["rng_sha256"] == y["rng_sha256"] for x, y in zip(ra["steps"], rb["steps"])],
             "artifacts": {}}
    for name in ("gradients-1.pt", "gradients-2.pt", "gradients-3.pt", "checkpoint.pt"):
        x, y = [torch.load(p / name, weights_only=False, map_location="cpu") for p in (a, b)]
        if name == "checkpoint.pt":
            x, y = x["model"], y["model"]
        rows = []
        for key in x:
            if x[key] is None:
                assert y[key] is None
                continue
            if not x[key].is_floating_point():
                assert torch.equal(x[key], y[key]), key
                continue
            delta = (x[key].double() - y[key].double()).abs()
            bad = (delta > 1e-5 + 1e-4 * x[key].double().abs()).sum().item()
            rows.append({"name": key, "max_abs": float(delta.max()) if delta.numel() else 0.,
                         "out_of_tolerance": bad, "elements": delta.numel()})
        entry["artifacts"][name] = {"max_abs": max([r["max_abs"] for r in rows], default=0.),
                                    "out_of_tolerance": sum(r["out_of_tolerance"] for r in rows),
                                    "elements": sum(r["elements"] for r in rows),
                                    "largest": sorted(rows, key=lambda r: r["max_abs"], reverse=True)[:5]}
        del x, y
    result[model] = entry
target = root / "training-deltas.json"
target.write_text(json.dumps(result, indent=2, sort_keys=True))
print(json.dumps(result, indent=2, sort_keys=True))
