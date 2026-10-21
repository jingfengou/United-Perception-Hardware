"""Audit and package an immutable completed cohort using its executed tool snapshot."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil


def module(directory, name):
    spec = importlib.util.spec_from_file_location(name, directory / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def copy_exact(source, destination, sha256):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise ValueError("Refusing to replace evidence: " + str(destination))
    else:
        shutil.copyfile(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--executed-tools", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--backend", choices=("nvidia", "metax"), required=True)
    args = parser.parse_args()
    root, results = args.root.resolve(), args.results.resolve()
    results.relative_to(root)
    tools = args.executed_tools.resolve()
    suite = module(tools, "run_cross_vendor_accuracy")
    audit = module(tools, "audit_cross_vendor_accuracy")
    metrics = module(tools, "recompute_cross_vendor_metrics")
    collector = module(tools, "collect_cross_vendor_evidence")
    plan = suite.load_plan(args.plan)
    if (root / (args.backend + "-evidence.tar.gz")).exists():
        raise ValueError("Evidence already packaged; use a fresh destination")
    canonical = root / "results"
    if canonical.exists():
        if canonical.resolve() != results:
            raise ValueError("Wrong canonical results directory")
    else:
        canonical.symlink_to(results, target_is_directory=True)
    for relative in ("data.json", "integrity.json", "cifar10/train.txt", "cifar10/test.txt",
                     "pennfudan/train.json", "pennfudan/test.json"):
        copy_exact(args.data_root / relative, root / "data" / relative, suite.sha256)
    audits, recomputed = {}, {}
    for model, spec in plan["models"].items():
        for seed in spec["seeds"]:
            path = canonical / (model + "-seed" + str(seed))
            audits[path.name] = audit.audit(path)
            recomputed[path.name] = metrics.reevaluate(path, args.data_root)
    (root / "checkpoint-audits.json").write_text(json.dumps(audits, indent=2, sort_keys=True))
    (root / "metric-reevaluations.json").write_text(json.dumps(recomputed, indent=2, sort_keys=True))
    result = collector.collect(root, args.plan, args.backend, root / (args.backend + "-evidence.tar.gz"))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
