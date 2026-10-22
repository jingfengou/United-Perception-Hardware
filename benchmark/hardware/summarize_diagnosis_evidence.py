"""Verify published cohort archives and build compact, auditable JSON/CSV views."""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
import tarfile
import xml.etree.ElementTree as ET


def sha(value):
    return hashlib.sha256(value).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_cohort(directory, backend):
    archive = directory / (backend + "-evidence.tar.gz")
    index = json.loads(archive.with_name(archive.name + ".index.json").read_text())
    if sha(archive.read_bytes()) != index["archive_sha256"]:
        raise ValueError("Archive hash mismatch")
    reports, environment, tests, seen = {}, None, None, set()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            path = PurePosixPath(member.name)
            if (not member.isfile() or not path.parts or path.parts[0] != backend or ".." in path.parts
                    or "\\" in member.name or any(":" in part for part in path.parts)
                    or path.suffix not in (".json", ".xml", ".log", ".txt") or member.size > 64 * 1024 * 1024):
                raise ValueError("Unexpected evidence member: " + member.name)
            relative = path.relative_to(backend).as_posix()
            content = stream.extractfile(member).read()
            if relative in seen or index["files"].get(relative) != sha(content):
                raise ValueError("Member hash mismatch: " + relative)
            seen.add(relative)
            if relative.startswith("results/") and path.name == "report.json":
                report = json.loads(content)
                if report["status"] != "completed" or report["backend"] != backend:
                    raise ValueError("Incomplete or misidentified run")
                audit = report["batch_audit"]
                if (len(audit) != report["training_steps"] or
                        sha(json.dumps([row["batch_sha256"] for row in audit]).encode()) != report["all_training_batches_sha256"]):
                    raise ValueError("Incomplete batch audit")
                reports[path.parent.name] = report
                keys = ("status", "model", "seed", "epochs", "protocol", "training_steps", "final_metrics",
                        "restored_metrics", "environment", "initial_state_sha256", "final_state_sha256",
                        "all_training_batches_sha256", "driver_sha256", "normalized_source_manifest_sha256")
                view = {key: report[key] for key in keys}
                view.update(raw_report={"archive": archive.name, "member": member.name, "sha256": sha(content)},
                            losses_sha256=sha(json.dumps(report["losses"]).encode()),
                            learning_rates_sha256=sha(json.dumps(report["learning_rates"]).encode()),
                            batch_audit_entries=len(audit))
                save(directory / backend / path.parent.relative_to(backend) / "run-summary.json", view)
            elif path.name in ("effective_config.json", "checkpoint-audit.json", "metric-reevaluation.json", "environment.json"):
                parsed = json.loads(content)
                if path.name in ("checkpoint-audit.json", "metric-reevaluation.json") and parsed["status"] != "passed":
                    raise ValueError("Failed audit")
                save(directory / backend / relative, parsed)
                if relative == "environment.json":
                    environment = parsed
            elif relative == "contracts-final.xml":
                xml = ET.fromstring(content)
                suites = list(xml.iter("testsuite"))
                if any(int(s.get(k, 0)) for s in suites for k in ("errors", "failures", "skipped")):
                    raise ValueError("Unsuccessful contract tests")
                tests = sum(int(s.get("tests", 0)) for s in suites)
                target = directory / backend / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
    if seen != index["files"].keys() or len(reports) != index["completed_runs"]:
        raise ValueError("Incomplete archive")
    return reports, environment, {"archive_sha256": index["archive_sha256"], "members": len(seen), "tests": tests}


def source_manifest(repository, revision=None):
    if revision:
        paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision, "up"], cwd=repository).decode().splitlines()
        paths = [p for p in paths if p.endswith(".py")]
        content = lambda p: subprocess.check_output(["git", "show", revision + ":" + p], cwd=repository)
    else:
        paths = sorted(p.relative_to(repository).as_posix() for p in (repository / "up").rglob("*.py"))
        content = lambda p: (repository / p).read_bytes()
    hashes = {p: sha(content(p).replace(b"\r\n", b"\n")) for p in paths}
    return sha(json.dumps(hashes, sort_keys=True).encode())


def diagnostics(root, directory, native_cohorts):
    load = lambda relative: json.loads((directory / relative).read_text())
    nv = load("raw/nvidia210/checkpoint-eval/summary.json")["cases"]
    mx = load("raw/metax210/checkpoint-eval-metax-origin/summary.json")["cases"]
    mx.update(load("raw/metax210/checkpoint-eval-nvidia-origin/summary.json")["cases"])
    swaps = []
    for origin in ("nvidia", "metax"):
        for model in ("yolov5s", "detr", "vit_b16"):
            prefix = origin + "-" + model + "-on-"
            a, b = nv[prefix + "nvidia"], mx[prefix + "metax"]
            training = native_cohorts[origin][model + "-seed131"]
            if (a["model_state_sha256"] != b["model_state_sha256"] or
                    a["model_state_sha256"] != training["final_state_sha256"] or
                    a["checkpoint_sha256"] != b["checkpoint_sha256"] or
                    a["metrics"] != b["metrics"] or a["metrics"] != training["final_metrics"]):
                raise ValueError("Shared-checkpoint inference differs")
            swaps.append({"origin": origin, "model": model, "seed": 131,
                          "model_state_sha256": a["model_state_sha256"], "metrics_identical": True,
                          "metrics": a["metrics"]})
    paired = load("comparisons/vit-paired.json")
    native_a = load("raw/vit-masks/report.json")
    native_b = load("raw/metax210/vit-masks/report.json")
    original = load("metax-no-dnn-original131.json")
    repeat = load("metax-no-dnn-repeat131.json")
    for name in ("repeat-checkpoint-audit.json", "repeat-metric-reevaluation.json"):
        items = list(load(name).values())
        if (len(items) != 1 or items[0]["status"] != "passed" or
                items[0]["report_sha256"] != sha((directory / "metax-no-dnn-repeat131.json").read_bytes())):
            raise ValueError("Missing or stale full-repeat audit")
    first = next((i for i, values in enumerate(zip(original["losses"], repeat["losses"], strict=True))
                  if values[0] != values[1]), None)
    repeated = {"same_initial_state": original["initial_state_sha256"] == repeat["initial_state_sha256"],
                "all_training_inputs_equal": original["all_training_batches_sha256"] == repeat["all_training_batches_sha256"],
                "losses_bitwise_equal": original["losses"] == repeat["losses"],
                "final_state_bitwise_equal": original["final_state_sha256"] == repeat["final_state_sha256"],
                "first_different_loss_step_zero_based": first,
                "comparison": load("repeat131-comparison.json"),
                "extra_complete_training_runs": 1, "extra_optimizer_steps": repeat["training_steps"]}
    nv_primitive = load("raw/nvidia210/torch-repeatability.json")
    mx_primitive = load("torch-repeatability-native5-final.json")
    disabled = load("torch-repeatability-no-dnn.json")
    for control in (mx_primitive, disabled):
        case = control["cases"]["conv2d"]
        reference = nv_primitive["cases"]["conv2d"]
        if case["input_hashes"] != reference["input_hashes"] or case["upstream_hash"] != reference["upstream_hash"]:
            raise ValueError("Different primitive control inputs")
    if mx_primitive["visible_devices"] != disabled["visible_devices"]:
        raise ValueError("DNN ablation used different physical device identifiers")
    scalar = lambda report: max(row["max_abs_vs_first"][2] for row in report["cases"]["conv2d"]["rows"])
    result = {"scope": "Numerical diagnosis only, not full-benchmark equivalence or a unique attribution of every AP point",
              "shared_checkpoint_inference": {"pairs": swaps, "pairs_with_identical_metrics": len(swaps),
                                              "coverage": "Both seed131 weight origins, all fixed held-out images"},
              "vit_masks": {"native_step0_loss_abs_difference": abs(native_a["steps"][0]["loss"] - native_b["steps"][0]["loss"]),
                            "paired_step_loss_abs_differences": [abs(row["losses"][0] - row["losses"][1]) for row in paired["steps"]],
                            "full_training_acceptance": False},
              "conv_weight_gradient_repeatability": {
                  "nvidia_native_max_abs_vs_first": scalar(nv_primitive),
                  "metax_native_max_abs_vs_first": scalar(mx_primitive),
                  "metax_dnn_disabled_max_abs_vs_first": scalar(disabled),
                  "repetitions": mx_primitive["repetitions"], "same_inputs_and_upstream_gradient": True,
                  "same_metax_physical_device": mx_primitive["visible_devices"],
                  "does_not_establish_full_training_determinism": True},
              "same_metax_full_training_repeat_dnn_disabled": repeated,
              "ops": load("comparisons/ops.json"),
              "nvidia_native_vs_portable_first_four_steps": load("yolo28-native-vs-portable.json")}
    save(root / "diagnostics/summary.json", result)
    candidates = {path.relative_to(directory).as_posix(): path for path in (directory / "raw").rglob("*.json")}
    candidates.update({path.relative_to(directory).as_posix(): path for path in (directory / "comparisons").glob("*.json")})
    for name in ("torch-repeatability-native5-final.json", "torch-repeatability-no-dnn.json", "torch-repeatability-isolated.json",
                 "repeat131-comparison.json", "metax-no-dnn-original131.json", "metax-no-dnn-repeat131.json",
                 "repeat-checkpoint-audit.json", "repeat-metric-reevaluation.json", "yolo28-native-vs-portable.json",
                 "yolo28-vs-metax.json", "detr28-vs-metax.json", "vit28-vs-metax.json", "vit210-masks-vs-metax.json"):
        if (directory / name).exists():
            candidates[name] = directory / name
    archive = root / "diagnostics/evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, path in sorted(candidates.items()):
            stream.add(path, arcname=name, recursive=False)
    save(root / "diagnostics/evidence.index.json", {"archive_sha256": sha(archive.read_bytes()),
         "files": {name: sha(path.read_bytes()) for name, path in sorted(candidates.items())},
         "omitted_tensor_payloads": {p.relative_to(directory).as_posix(): {"sha256": sha(p.read_bytes()), "bytes": p.stat().st_size}
                                    for p in (directory / "raw").rglob("tensors.pt")},
         "note": "Tensor payloads retained in the task workspace and on both servers; no model weights or private keys published"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    coverage = {"status": "passed", "scope": "Two separately declared policies, never pooled for equivalence",
                "phases": {}, "complete_training_runs": 0, "optimizer_steps": 0}
    manifests = {"native": source_manifest(args.repository, "1d891db70679e0f26ff9f985460e807a5fdce78b"),
                 "no-dnn": source_manifest(args.repository)}
    all_cohorts = {}
    for phase, plan_name in (("native", "plan.json"), ("no-dnn", "plan-no-dnn.json")):
        directory = args.root / phase
        plan = json.loads((args.root / plan_name).read_text())
        summary = json.loads((directory / "summary.json").read_text())
        if summary["plan_sha256"] != sha((args.root / plan_name).read_bytes()):
            raise ValueError("Statistics use a different plan")
        rows, environments, cohorts, metadata = [], {}, {}, {}
        for backend in ("nvidia", "metax"):
            cohorts[backend], environments[backend], metadata[backend] = read_cohort(directory, backend)
            for report in cohorts[backend].values():
                if report["normalized_source_manifest_sha256"] != manifests[phase]:
                    raise ValueError("Executed application source differs from declared Git/working source")
        for model, spec in plan["models"].items():
            deltas = []
            for pair in summary["models"][model]["pairs"]:
                run = model + "-seed" + str(pair["seed"])
                a, b = [cohorts[backend][run] for backend in ("nvidia", "metax")]
                for key in ("model", "seed", "epochs", "protocol", "training_steps", *plan["pairing"]):
                    if a[key] != b[key]:
                        raise ValueError("Unfair pair: " + key)
                for report, digest in zip((a, b), pair["report_sha256"], strict=True):
                    backend = report["backend"]
                    expected = json.loads((directory / (backend + "-evidence.tar.gz.index.json")).read_text())["files"]["results/" + run + "/report.json"]
                    if digest != expected or report["training_steps"] != spec["expected_steps_per_run"]:
                        raise ValueError("Stale statistics or incomplete schedule")
                values = [r["final_metrics"][spec["primary_metric"]] * spec["metric_to_percentage_points"] for r in (a, b)]
                delta = values[1] - values[0]
                if delta != pair["delta_percentage_points"]:
                    raise ValueError("Incorrect reported metric delta")
                deltas.append(delta)
                rows.append({"model": model, "seed": pair["seed"], "nvidia": values[0], "metax": values[1],
                             "delta_pp": delta, "all_training_batches_equal": True,
                             "steps_per_vendor": a["training_steps"]})
            if abs(statistics.mean(deltas) - summary["models"][model]["equivalence_test"]["mean_delta_percentage_points"]) > 1e-12:
                raise ValueError("Incorrect mean")
        with (directory / "pairs.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        count = sum(len(values) for values in cohorts.values())
        steps = sum(r["training_steps"] for values in cohorts.values() for r in values.values())
        coverage["complete_training_runs"] += count
        coverage["optimizer_steps"] += steps
        coverage["phases"][phase] = {"runs": count, "steps": steps, "fair_pairs": len(rows),
                                     "all_training_inputs_equal": True, "archives": metadata,
                                     "application_source_manifest": manifests[phase]}
        all_cohorts[phase] = cohorts
        if phase == "native":
            canonical = lambda n: re.sub("[-_.]+", "-", n).lower()
            packages = {b: {canonical(n): v for n, v in e["packages"].items()} for b, e in environments.items()}
            checked = {}
            for line in (args.repository / "requirements/cross-vendor-2.10-nvidia.txt").read_text().splitlines():
                if line.startswith("#") or "==" not in line:
                    continue
                name, version = line.split("==")
                if name in ("torch", "torchvision"):
                    continue
                observed = {b: p.get(canonical(name)) for b, p in packages.items()}
                if any(value != version for value in observed.values()):
                    raise ValueError("Dependency alignment mismatch: " + name + repr(observed))
                checked[name] = {"expected": version, "observed": observed}
            save(args.root / "aligned-packages.json", {"status": "passed", "shared_application_packages": checked,
                 "python_patch_versions_equal": False, "vendor_builds_identical": False})
    save(args.root / "coverage.json", coverage)
    if args.diagnostics:
        diagnostics(args.root, args.diagnostics, all_cohorts["native"])
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
