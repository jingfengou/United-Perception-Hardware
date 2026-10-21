"""Audit one complete vendor cohort without claiming cross-vendor equivalence."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import statistics


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def summarize(root, backend, repository):
    destination = root / (backend + "-completion.json")
    if destination.exists():
        raise ValueError("Preserve the existing vendor completion record")
    import numpy
    import scipy

    suite = sibling("run_cross_vendor_accuracy")
    reader = sibling("summarize_diagnosis_evidence")
    sources = sibling("summarize_portable_cohort")
    plan = suite.load_plan(root / "plan.json")
    manifest, source_files, snapshot, _ = sources.read_source(root, backend)
    expected = plan.get("expected_normalized_up_manifest_sha256")
    if expected and manifest != expected:
        raise ValueError("Source snapshot differs from the frozen plan")
    reports, environment, archive = reader.read_cohort(root, backend)
    index = json.loads((root / (backend + "-evidence.tar.gz.index.json")).read_text())
    if index["plan_sha256"] != suite.sha256(root / "plan.json"):
        raise ValueError("Evidence uses another plan")
    names = {model + "-seed" + str(seed) for model, spec in plan["models"].items() for seed in spec["seeds"]}
    if reports.keys() != names:
        raise ValueError("Missing or unexpected complete runs")
    models, steps = {}, 0
    for model, spec in plan["models"].items():
        rows, cohort = [], {}
        for seed in spec["seeds"]:
            name = model + "-seed" + str(seed)
            report = reports[name]
            suite.verify_run(report, plan, model, seed, backend, index["data_sha256"],
                             snapshot["files"]["tools/validate_training_accuracy.py"])
            suite.SUITE.verify_cohort(cohort, backend, report)
            if report["source_files"] != source_files or report["normalized_source_manifest_sha256"] != manifest:
                raise ValueError("Training source and collected snapshot differ")
            adapter = snapshot["files"]["tools/accuracy_detr_adapter.py"] if model == "detr" else None
            if report["adapter_sha256"] != adapter:
                raise ValueError("Changed external model adapter")
            metric = report["final_metrics"][spec["primary_metric"]] * spec["metric_to_percentage_points"]
            rows.append({"seed": seed, "training_steps": report["training_steps"], "metric_percent": metric,
                         "final_state_sha256": report["final_state_sha256"],
                         "report_sha256": index["files"]["results/" + name + "/report.json"],
                         "checkpoint_audit": "passed", "independent_metric_reevaluation": "passed",
                         "numerics": report["numerics"]})
            steps += report["training_steps"]
            view_path = root / backend / "results" / name / "run-summary.json"
            view = json.loads(view_path.read_text())
            view["numerics"] = report["numerics"]
            sources.save(view_path, view)
        values = [row["metric_percent"] for row in rows]
        models[model] = {"metric": spec["primary_metric"], "mean_percent": statistics.mean(values),
                         "std_percent": statistics.stdev(values), "runs": rows}
    result = {"status": "completed_single_vendor_cohort",
              "scope": "Complete declared budgets and artifact integrity, not cross-vendor accuracy acceptance",
              "cross_vendor_equivalence": "not_assessed_by_single_vendor_summary", "backend": backend,
              "plan_sha256": suite.sha256(root / "plan.json"), "data_sha256": index["data_sha256"],
              "complete_runs": len(reports), "optimizer_steps": steps, "models": models,
              "source_manifest": manifest, "source_snapshot_sha256": snapshot["archive_sha256"],
              "matches_current_working_up": manifest == reader.source_manifest(repository),
              "environment": environment, "evidence": archive,
              "postprocessing": {"python": platform.python_version(), "numpy": numpy.__version__,
                                 "scipy": scipy.__version__},
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    sources.save(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backend", choices=("nvidia", "metax"), required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = summarize(args.root, args.backend, args.repository)
    print(json.dumps({key: result[key] for key in ("status", "backend", "complete_runs", "optimizer_steps")}))


if __name__ == "__main__":
    main()
