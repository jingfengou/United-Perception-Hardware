import json
from pathlib import Path
import sys
import torch

torch.set_num_threads(2)
root = Path(sys.argv[1])


def differences(first, second):
    rows = []
    def visit(a, b, name):
        if isinstance(a, torch.Tensor):
            assert a.shape == b.shape and a.dtype == b.dtype, name
            if not a.is_floating_point():
                assert torch.equal(a, b), name
                return
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
            delta = (a.double() - b.double()).abs()
            rows.append({"name": name, "elements": a.numel(), "exact": torch.equal(a, b),
                         "max_abs": float(delta.max()) if delta.numel() else 0.,
                         "outside_tolerance": int((delta > 1e-5 + 1e-4 * a.double().abs()).sum())})
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                visit(a[key], b[key], name + "/" + str(key))
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, name + "/" + str(i))
        else:
            assert a == b, name
    visit(first, second, "root")
    return {"max_abs": max([row["max_abs"] for row in rows], default=0.),
            "outside_tolerance": sum(row["outside_tolerance"] for row in rows),
            "elements": sum(row["elements"] for row in rows),
            "tensor_count": len(rows), "exact_tensor_count": sum(row["exact"] for row in rows),
            "largest": sorted(rows, key=lambda row: row["max_abs"], reverse=True)[:5]}


result = {"scope": "same-vendor, same-seed repeat controls, not multiseed accuracy",
          "seed": 131, "steps": 3, "rtol": 1e-4, "atol": 1e-5, "models": {}}
for model in ("resnet18", "yolov5s", "detr", "vit_b16"):
    before = root / "run-01/baseline" / model / "training"
    changed = root / "run-01/changed" / model / "training"
    repeat = root / "repeat-01" / model
    reports = [json.loads((path / "report.json").read_text()) for path in (before, changed, repeat)]
    assert all(report["passed"] for report in reports)
    row = {}
    for label, path, report in (("pre_vs_post", changed, reports[1]), ("pre_vs_pre_repeat", repeat, reports[2])):
        original = reports[0]
        assert original["environment"] == report["environment"]
        assert original["initial_state_sha256"] == report["initial_state_sha256"]
        assert [step["batch_sha256"] for step in original["steps"]] == [step["batch_sha256"] for step in report["steps"]]
        assert [step["rng_sha256"] for step in original["steps"]] == [step["rng_sha256"] for step in report["steps"]]
        entry = {"initial_state_batches_and_rng_exact": True,
                 "loss_deltas": [b["loss"] - a["loss"] for a, b in zip(original["steps"], report["steps"])],
                 "artifacts": {}}
        for name in ("gradients-1.pt", "gradients-2.pt", "gradients-3.pt", "checkpoint.pt", "eval-1.pt", "eval-2.pt"):
            a, b = [torch.load(folder / name, map_location="cpu", weights_only=False) for folder in (before, path)]
            if name == "checkpoint.pt":
                a, b = a["model"], b["model"]
            entry["artifacts"][name] = differences(a, b)
            del a, b
        row[label] = entry
    result["models"][model] = row
    (root / "repeat-controls.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("audited", model, flush=True)
