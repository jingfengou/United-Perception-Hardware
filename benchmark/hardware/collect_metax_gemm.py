"""Fail-closed audit for the single-seed DETR GEMM fix, including raw evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile

from diagnose_detr_origin import compare_gradients, digest, tensor_comparison, write_json
from prepare_cross_vendor_data import verify_data
from validate_platform_equivalence import file_hash, source_manifest

PROFILE = "metax_deterministic_v2"
TOLERANCES = {"rtol": 1e-4, "atol": 1e-5, "metric_atol": 1e-6}
CAPTURES = ("baseline-full", "changed-full-0", "changed-full-1")


def compare_reports(a, b):
    for report in (a, b):
        assert report["completed"] and len(report["steps"]) == 720
        assert report["profile"] == PROFILE and (report["rtol"], report["atol"]) == (1e-4, 1e-5)
        assert all(math.isfinite(row["loss"]) for row in report["steps"])
    assert a["initial_model_sha256"] == b["initial_model_sha256"]
    assert (a["torch"], a["gpu"]) == (b["torch"], b["gpu"])
    assert all(x["input_sha256"] == y["input_sha256"] for x, y in zip(a["steps"], b["steps"], strict=True))
    fields = ("raw_gradient_sha256", "clipped_gradient_sha256", "weight_sha256", "loss")
    result = {"steps": 720, "inputs_equal": True, "initial_weights_equal": True,
              "exact": {key: all(x[key] == y[key] for x, y in zip(a["steps"], b["steps"], strict=True)) for key in fields},
              "maximum_loss_absolute_difference": max(abs(x["loss"] - y["loss"]) for x, y in zip(a["steps"], b["steps"], strict=True))}
    result["passed"] = all(result["exact"].values())
    return result


def audit(root):
    import torch
    read = lambda name: json.loads((root / name).read_text())
    stage = read("stage-source.json")
    assert file_hash(root / "data.json") == stage["data_sha256"]
    data_verified = verify_data(read("data.json"))
    for key, folder in (("source", "candidate-v2"), ("baseline", "baseline-v2")):
        assert stage[key] == source_manifest(root / folder), "Frozen source changed"
    for folder in ("up/models/", "up/tasks/", "configs/"):
        original = {k: v for k, v in stage["baseline"]["normalized_files"].items() if k.startswith(folder)}
        changed = {k: v for k, v in stage["source"]["normalized_files"].items() if k.startswith(folder)}
        assert original == changed, "Model/config source changed"
    reports = {name: read(name + "/report.json") for name in (*CAPTURES, "official-full")}
    result = {"scope": "C500, one seed 131, 12 epochs / 720 updates per run; not a cross-vendor accuracy claim",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "tolerances": TOLERANCES,
              "source": {key: stage[key]["sha256"] for key in ("baseline", "source")},
              "data_sha256": stage["data_sha256"], "verified_data": data_verified,
              "comparisons": {}, "runs": {}, "primitive": {}, "inference": {}}
    result["prior_strategy_control"] = read("frozen-protocol-control.json")
    assert all(result["prior_strategy_control"][key] is True for key in (
        "prior_v1_initial_weights_equal", "all_720_actual_inputs_equal", "all_720_learning_rates_equal"))
    reference = reports["changed-full-0"]
    for name, report in reports.items():
        pair = compare_reports(reference, report)
        states = [torch.load(root / folder / "final.pt", map_location="cpu", weights_only=False)
                  for folder in ("changed-full-0", name)]
        final = compare_gradients(states[0]["model"], states[1]["model"])
        for state in states:
            for value in state["model"].values():
                assert torch.isfinite(value).all()
            for entry in state["optimizer"]["state"].values():
                for value in entry.values():
                    if isinstance(value, torch.Tensor):
                        assert torch.isfinite(value).all()
        assert digest(states[1]["model"]) == report["steps"][-1]["weight_sha256"]
        pair["final_weights"] = {"exact": final["exact"], "outside_tolerance": final["outside_tolerance"]}
        pair["optimizer_exact"] = digest(states[0]["optimizer"]) == digest(states[1]["optimizer"])
        pair["passed"] &= final["exact"] and pair["optimizer_exact"]
        result["comparisons"][name] = pair
        result["runs"][name] = {"training_steps": len(report["steps"]), "model_finite": True,
                               "optimizer_finite": True, "final_model_sha256": digest(states[1]["model"]),
                               "driver_sha256": report["driver_sha256"]}
        if name in CAPTURES:
            assert report["audit_completed"] and report["checkpoint_restore"]
            assert report["final_metrics"] == report["restored_metrics"]
            evaluations = report["evaluations"]
            assert len(evaluations) == 15 and all(row["purpose"] == "training" and row["training_state_unchanged"]
                                                 and row["module_modes_restored"] for row in evaluations)
            assert [row["metrics"] for row in evaluations] == [row["metrics"] for row in reference["evaluations"]]
            assert [row["predictions_sha256"] for row in evaluations] == [row["predictions_sha256"] for row in reference["evaluations"]]
            assert read(name + "/effective_config.json") == read("changed-full-0/effective_config.json")
            config = read(name + "/effective_config.json")
            assert config["runtime"]["random_seed"] == 131 and config["trainer"]["max_epoch"] == 12
            assert [row["lr"] for row in report["steps"]] == [row["lr"] for row in reference["steps"]]
            result["runs"][name].update(final_metrics=report["final_metrics"], checkpoint_restore=True,
                                        evaluations=len(evaluations), numerics=report["numerics"],
                                        elapsed_seconds=report["elapsed_seconds"])
        else:
            assert not report["up_imported"] and report["platform_imported"] and report["criterion"] == "matched"
    for index in range(2):
        name = "primitive-layout-" + str(index)
        report = read(name + "/report.json")
        assert report["passed"] and report["case"]["exact"] and report["case"]["repeats"] == 100
        assert report["case"]["reference"]["outside_tolerance"] == 0 and len(report["contracts"]) == 24
        assert all(row["strides_preserved"] for row in report["contracts"])
        assert (report["rtol"], report["atol"]) == (1e-4, 1e-5)
        result["primitive"][name] = {key: report[key] for key in ("input_sha256", "rtol", "atol", "torch", "gpu")}
        result["primitive"][name].update(reference=report["case"]["reference"], repeats=100, exact=True,
                                         contracts=len(report["contracts"]))
    primitive = [torch.load(root / name / "first-result.pt", weights_only=True) for name in result["primitive"]]
    result["primitive_cross_process"] = tensor_comparison(*primitive)
    assert result["primitive_cross_process"]["exact"]
    traces = [read(name + "/trace.json") for name in ("trace58-fixed-0", "trace58-fixed-1")]
    assert all(row["completed"] for row in traces) and traces[1]["first_difference"] is None
    assert traces[1]["operation_count_equal"] and traces[0]["gradient_sha256"] == traces[1]["gradient_sha256"]
    result["fixed_trace"] = {"step": 58, "operations": len(traces[0]["operations"]), "exact": True}
    for name in (*CAPTURES, "official-full"):
        report = read("inference/" + name + "/report.json")
        assert report["passed"] and report["purpose"] == "inference" and report["seen_images"] == 50
        assert not report["optimizer_or_scaler_created"] and report["predict_called"]
        assert all(report[key] == value for key, value in TOLERANCES.items())
        assert report["numerics"]["profile"] == PROFILE and report["metrics"] == reference["final_metrics"]
        result["inference"][name] = report
    assert not torch.cuda.is_initialized(), "The collector must remain a CPU-only audit"
    result["passed"] = all(row["passed"] for row in result["comparisons"].values())
    return result


def collect(root, output):
    index_path = Path(str(output) + ".index.json")
    if output.exists() or index_path.exists():
        raise FileExistsError("Preserve the existing evidence archive")
    report = audit(root)
    write_json(root / "summary.json", report)
    index = {"text_files": {}, "retained_binary_files": {}, "passed": report["passed"]}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if relative in ("collect.log", "postprocess.log", "postprocess-status.json"):
            continue
        if relative.startswith(("candidate-v1/", "candidate-v2/", "baseline-v2/")):
            continue
        category = ("retained_binary_files" if path.suffix in (".pt", ".pth", ".tar") else
                    "text_files" if path.suffix in (".json", ".py", ".log", ".txt", ".jsonl") else None)
        if category:
            index[category][relative] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    payload = json.dumps(index, indent=2, sort_keys=True).encode()
    with tarfile.open(output, "x:gz") as packed:
        for name in index["text_files"]:
            packed.add(root / name, arcname=name, recursive=False)
        info = tarfile.TarInfo("evidence-index.json")
        info.size = len(payload)
        packed.addfile(info, io.BytesIO(payload))
    with tarfile.open(output) as packed:
        assert len(packed.getmembers()) == len(index["text_files"]) + 1
        for member in packed.getmembers():
            value = packed.extractfile(member).read()
            if member.name == "evidence-index.json":
                assert value == payload
            else:
                record = index["text_files"][member.name]
                assert len(value) == record["bytes"] and hashlib.sha256(value).hexdigest() == record["sha256"]
    index["archive"] = {"name": output.name, "bytes": output.stat().st_size, "sha256": file_hash(output)}
    write_json(index_path, index)
    return index["archive"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.root.resolve(), args.output.resolve()), indent=2))
