"""Package complete vendor-cohort evidence without weights, data images or credentials."""
import argparse
import importlib.util
import json
from pathlib import Path
import tarfile


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = sibling("run_cross_vendor_accuracy")


def safe_relative(root, path):
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    if not path.is_file() or path.is_symlink():
        raise ValueError("Only regular, task-local evidence files are allowed: " + str(path))
    return path.relative_to(root).as_posix()


def collect(root, plan_path, backend, destination):
    plan = TOOL.load_plan(plan_path)
    if destination.exists():
        raise ValueError("Use a new evidence archive path; do not replace existing evidence")
    data_path = root / "data/data.json"
    data_hash = TOOL.sha256(data_path)
    driver_hash = TOOL.sha256(Path(__file__).with_name("validate_training_accuracy.py"))
    members = set()
    for model, spec in plan["models"].items():
        for seed in spec["seeds"]:
            run = root / "results" / (model + "-seed" + str(seed))
            report_path = run / "report.json"
            report = json.loads(report_path.read_text())
            TOOL.verify_run(report, plan, model, seed, backend, data_hash, driver_hash)
            audit_path = run / "checkpoint-audit.json"
            audit = json.loads(audit_path.read_text())
            if audit["status"] != "passed" or audit["report_sha256"] != TOOL.sha256(report_path):
                raise ValueError("Missing or stale checkpoint audit: " + str(run))
            if audit["checkpoint_sha256"] != TOOL.sha256(run / "comparison_state.pt"):
                raise ValueError("Checkpoint changed since its audit: " + str(run))
            reeval_path = run / "metric-reevaluation.json"
            reeval = json.loads(reeval_path.read_text())
            if (reeval["status"] != "passed" or reeval["report_sha256"] != TOOL.sha256(report_path)
                    or reeval["predictions_sha256"] != TOOL.sha256(run / "predictions/results.txt")):
                raise ValueError("Missing or stale independent metric reevaluation: " + str(run))
            members.update((report_path, run / "effective_config.json", audit_path, reeval_path,
                            root / "results" / (run.name + ".log")))
            predictions = sorted((run / "predictions").rglob("*.txt"))
            if not predictions:
                raise ValueError("Missing final validation predictions: " + str(run))
            members.update(predictions)
    members.update((root / "environment.json", root / "contracts-final.xml", data_path, root / "data/integrity.json"))
    for name in ("cifar10", "pennfudan"):
        for split in ("train", "test"):
            members.add(root / "data" / name / (split + (".txt" if name == "cifar10" else ".json")))
    members.update((root / "results").glob("execution-*.json"))
    members.update(root.glob("training*.log"))
    for path in (root / "preflight").rglob("*"):
        if path.is_file() and (path.suffix == ".log" or path.name in ("report.json", "effective_config.json")):
            members.add(path)
    manifest = {safe_relative(root, path): TOOL.sha256(path) for path in sorted(members)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for name in manifest:
            archive.add(root / name, arcname=backend + "/" + name, recursive=False)
    index = {"backend": backend, "plan_sha256": TOOL.sha256(plan_path), "data_sha256": data_hash,
             "collector_sha256": TOOL.sha256(__file__), "archive_sha256": TOOL.sha256(destination),
             "completed_runs": sum(len(spec["seeds"]) for spec in plan["models"].values()),
             "files": manifest, "exclusions": ["checkpoints", "images", "private keys", "wheels", "SDK binaries"]}
    index_path = destination.with_name(destination.name + ".index.json")
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    return {"archive": str(destination), "index": str(index_path), "files": len(manifest),
            "archive_sha256": index["archive_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--backend", choices=("nvidia", "metax"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.root, args.plan, args.backend, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
