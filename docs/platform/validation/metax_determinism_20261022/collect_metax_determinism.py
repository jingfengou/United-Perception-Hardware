"""Audit and package the bounded C500 diagnosis, retaining failed controls."""
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import tarfile

from prepare_cross_vendor_data import verify_data
from validate_platform_equivalence import file_hash, source_manifest, write_json

MODELS = ("resnet18", "yolov5s", "detr")


def audit(root, baseline, source):
    sources = {"baseline": source_manifest(baseline), "changed": source_manifest(source)}
    original, changed = [value["normalized_files"] for value in sources.values()]
    models = [name for name in original if name.startswith("up/models/") or (
        name.startswith("up/tasks/") and "/models/" in name)]
    assert all(original[name] == changed.get(name) for name in models), "Model source changed"
    result = {"scope": "same C500, seed 131, three updates; no multiseed accuracy cohort",
              "rtol": 1e-4, "atol": 1e-5, "source_manifests": sources,
              "model_source_files_unchanged": len(models), "data": {}, "models": {}}
    for relative in ("data/data.json", "fixed-v1/data/data.json"):
        path = root / relative
        result["data"][relative] = {"sha256": file_hash(path), "verified": verify_data(json.loads(path.read_text()))}
    for model in MODELS:
        runs, training_reports = {}, {}
        for label in ("baseline", "changed-0", "changed-1"):
            for phase in ("training", "inference"):
                path = root / "fixed-v1" / model / label / phase / "report.json"
                report = json.loads(path.read_text())
                assert report["passed"], str(path)
                expected = sources["baseline" if label == "baseline" else "changed"]
                assert report["source"] == expected, "Source changed after execution: " + str(path)
                runs[label + "/" + phase] = {"report": path.relative_to(root).as_posix(), "passed": report["passed"],
                                              "adaptation": report["adaptation"]}
                if phase == "training":
                    training_reports[label] = report
        comparisons = {}
        for label in ("changed-0", "changed-1"):
            path = root / "fixed-v1" / model / ("compare-" + label) / "report.json"
            report = json.loads(path.read_text())
            assert (report["rtol"], report["atol"]) == (1e-4, 1e-5)
            comparisons[label] = report
        result["models"][model] = {"runs": runs, "comparisons": comparisons,
                                   "passed": all(value["passed"] for value in comparisons.values()),
                                   "losses_exact": all(
                                       [(row["loss"], row["losses"]) for row in report["steps"]] ==
                                       [(row["loss"], row["losses"]) for row in training_reports["baseline"]["steps"]]
                                       for report in training_reports.values()),
                                   "metrics_exact": all(
                                       [row["metrics"] for row in report["evaluations"]] ==
                                       [row["metrics"] for row in training_reports["baseline"]["evaluations"]]
                                       for report in training_reports.values())}
    result["passed"] = all(value["passed"] for value in result["models"].values())
    cases = {}
    for model in MODELS:
        paths = sorted((root / "candidate-replay" / model).glob("*/report.json"))
        assert len(paths) == 4, "Expected all four retained real cases: " + model
        cases[model] = []
        for path in paths:
            replay = json.loads(path.read_text())
            assert (replay["rtol"], replay["atol"]) == (1e-4, 1e-5)
            stats = replay["providers"]["metax_deterministic_v1"]
            assert stats["repeats"] == 10
            cases[model].append({"report": path.relative_to(root).as_posix(), "exact": stats["exact"],
                                 "native_parity": all(value["outside_tolerance"] == 0 for value in stats["vs_dnn_on"]),
                                 "max_abs_vs_native": [value["max_abs"] for value in stats["vs_dnn_on"]]})
    result["primitive_cases"] = cases
    result["passed"] &= all(case["exact"] and case["native_parity"] for entries in cases.values() for case in entries)
    return result


def collect(root, output, baseline, source):
    if output.exists() or Path(str(output) + ".index.json").exists():
        raise FileExistsError("Preserve the existing archive: " + str(output))
    report = audit(root, baseline, source)
    write_json(root / "audit.json", report)
    process = subprocess.run(["mx-smi"], capture_output=True, text=True, check=False)
    write_json(root / "environment.json", {"command": ["mx-smi"], "exit_code": process.returncode,
                                           "stdout": process.stdout, "stderr": process.stderr,
                                           "observed_at_utc": datetime.now(timezone.utc).isoformat()})
    index = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "text_files": {}, "retained_binary_files": {},
             "audit_passed": report["passed"], "root": str(root)}
    included = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith("source-v1/"):
            if relative not in {"source-v1/packages/up-platform/src/up_platform/backends/metax/determinism.py",
                                "source-v1/packages/up-platform/src/up_platform/backends/metax/device.py",
                                "source-v1/packages/up-platform/src/up_platform/__init__.py"}:
                continue
        if path.suffix in (".pt", ".pth", ".tar"):
            index["retained_binary_files"][relative] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
        elif path.suffix in (".json", ".py", ".log", ".txt"):
            index["text_files"][relative] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
            included.append(path)
    payload = json.dumps(index, indent=2, sort_keys=True).encode()
    with tarfile.open(output, "w:gz") as archive:
        for path in included:
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
        info = tarfile.TarInfo("evidence-index.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    index["archive"] = {"filename": output.name, "bytes": output.stat().st_size, "sha256": file_hash(output)}
    write_json(Path(str(output) + ".index.json"), index)
    return {"passed": report["passed"], "archive": index["archive"],
            "text_files": len(index["text_files"]), "retained_binary_files": len(index["retained_binary_files"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.root.resolve(), args.output.resolve(), args.baseline.resolve(), args.source.resolve()), indent=2))
