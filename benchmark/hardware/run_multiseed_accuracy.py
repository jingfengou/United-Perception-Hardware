"""Run pinned, paired whole-UP training schedules and test metric equivalence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)


def job_name(model, seed, variant, repeat=0):
    return "{}-seed{}-{}-repeat{}".format(model, seed, variant, repeat)


def equivalent(deltas, margin, alpha=.05):
    from scipy.stats import t

    if len(deltas) < 2 or not all(math.isfinite(value) for value in deltas):
        raise ValueError("At least two finite paired deltas are required")
    if not math.isfinite(margin) or margin <= 0 or not 0 < alpha < .5:
        raise ValueError("Invalid equivalence margin or alpha")
    mean = statistics.mean(deltas)
    deviation = statistics.stdev(deltas)
    error = deviation / math.sqrt(len(deltas))
    radius = float(t.ppf(1 - alpha, len(deltas) - 1)) * error
    descriptive_radius = float(t.ppf(.975, len(deltas) - 1)) * error
    if error:
        p_lower = float(t.sf((mean + margin) / error, len(deltas) - 1))
        p_upper = float(t.cdf((mean - margin) / error, len(deltas) - 1))
    else:
        p_lower = 0. if mean > -margin else (.5 if mean == -margin else 1.)
        p_upper = 0. if mean < margin else (.5 if mean == margin else 1.)
    return {"n": len(deltas), "paired_deltas_percentage_points": deltas,
            "mean_delta_percentage_points": mean, "std_delta_percentage_points": deviation,
            "confidence_interval_90": [mean - radius, mean + radius],
            "descriptive_confidence_interval_95": [mean - descriptive_radius, mean + descriptive_radius],
            "margin_percentage_points": margin, "alpha": alpha,
            "tost_p_lower": p_lower, "tost_p_upper": p_upper,
            "tost_p_value": max(p_lower, p_upper),
            "equivalent_within_margin": p_lower < alpha and p_upper < alpha,
            "interpretation": "Fixed-holdout mean over seeds only; not exact identity or full-benchmark equivalence"}


def verify_report(report, spec, seed, data_hash, driver_hash, protocol, expected_model=None):
    if expected_model is not None and report["model"] != expected_model:
        raise ValueError("Run contains a different model")
    if report["status"] != "completed" or report["seed"] != seed or report["epochs"] != spec["epochs"]:
        raise ValueError("Incomplete or mismatched training run")
    if report["data_manifest_sha256"] != data_hash or report["driver_sha256"] != driver_hash:
        raise ValueError("Run data or driver changed; refusing accidental reuse")
    if report["protocol"] != protocol:
        raise ValueError("Run protocol differs")
    if not report["training_steps"] or report["training_steps"] != report["expected_training_steps"]:
        raise ValueError("Training budget is incomplete")
    if "expected_steps_per_run" in spec and report["training_steps"] != spec["expected_steps_per_run"]:
        raise ValueError("Training budget differs from the predeclared plan")
    if len(report["losses"]) != report["training_steps"] or len(report["learning_rates"]) != report["training_steps"]:
        raise ValueError("Loss or scheduler trace is incomplete")
    if not all(math.isfinite(value) for value in report["losses"]):
        raise ValueError("Non-finite loss")
    if report["final_metrics"] != report["restored_metrics"]:
        raise ValueError("Checkpoint restoration changed metrics")
    if not math.isfinite(report["final_metrics"][spec["primary_metric"]]):
        raise ValueError("Non-finite primary metric")


def verify_cohort(cohorts, variant, report):
    keys = ("data_manifest_sha256", "driver_sha256", "protocol", "source_manifest_sha256",
            "native_binaries", "adapter_sha256", "external_model")
    signature = {key: report.get(key) for key in keys}
    signature["environment"] = {key: value for key, value in report["environment"].items()
                                if key != "visible_devices"}
    if variant in cohorts and cohorts[variant] != signature:
        raise ValueError("Mixed source, data or software versions across seeds: " + variant)
    cohorts[variant] = signature


def ensure_device_idle(device):
    processes = subprocess.check_output([
        "nvidia-smi", "-i", device, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True).strip()
    if processes:
        raise RuntimeError("GPU has active compute processes; will not interrupt them: " + device)
    memory = subprocess.check_output([
        "nvidia-smi", "-i", device, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True).strip()
    if int(memory) > 512:
        raise RuntimeError("GPU is in use; will not interrupt its owner: " + device)


def load_plan(path):
    plan = json.loads(path.read_text())
    if plan["protocol"] not in ("multiseed_v1", "multiseed_v2"):
        raise ValueError("Unsupported suite protocol")
    for spec in plan["models"].values():
        if len(spec["seeds"]) < 2 or len(set(spec["seeds"])) != len(spec["seeds"]):
            raise ValueError("Need distinct predeclared seeds")
        if not set(spec["repeat_seeds"]).issubset(spec["seeds"]) or spec["epochs"] <= 0:
            raise ValueError("Invalid repeat seeds or epoch budget")
    return plan


def run_suite(args):
    plan = load_plan(args.plan)
    args.output.mkdir(parents=True, exist_ok=True)
    driver = Path(__file__).with_name("validate_training_accuracy.py")
    driver_hash, data_hash = sha256(driver), sha256(args.data)
    selected = args.models or list(plan["models"])
    devices = args.devices.split(",")
    if len(set(devices)) != len(devices):
        raise ValueError("Each worker needs a separate device")
    manifest = {"plan_sha256": sha256(args.plan), "data_sha256": data_hash,
                "driver_sha256": driver_hash, "suite_sha256": sha256(__file__),
                "baseline": str(args.baseline.resolve()), "routed": str(args.routed.resolve()),
                "data": str(args.data.resolve()), "devices": devices, "models": selected}
    manifest_path = args.output / ("execution-" + "-".join(selected) + ".json")
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Execution manifest changed; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    groups = [(model, seed) for model in selected for seed in plan["models"][model]["seeds"]]
    stop = threading.Event()

    def worker(index, device):
        for group_index, (model, seed) in enumerate(groups):
            if group_index % len(devices) != index:
                continue
            spec = plan["models"][model]
            for repeat in ([0, 1] if seed in spec["repeat_seeds"] else [0]):
                variants = ["baseline", "routed"] if seed % 2 else ["routed", "baseline"]
                for variant in variants:
                    if stop.is_set():
                        return
                    name = job_name(model, seed, variant, repeat)
                    target = args.output / name
                    report_path = target / "report.json"
                    if report_path.exists():
                        report = json.loads(report_path.read_text())
                        verify_report(report, spec, seed, data_hash, driver_hash, plan["protocol"], model)
                        if report["environment"]["visible_devices"] != device:
                            raise ValueError("Resumed run used a different GPU")
                        emit({"event": "verified_completed_run", "job": name})
                        continue
                    if target.exists() and any(target.iterdir()):
                        raise ValueError("Partial run needs inspection, not silent deletion: " + str(target))
                    ensure_device_idle(device)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, OMP_NUM_THREADS="2",
                               MKL_NUM_THREADS="2", PYTHONHASHSEED="0")
                    if args.compat:
                        env["PYTHONPATH"] = str(args.compat.resolve())
                    command = [sys.executable, str(driver), "--source", str(getattr(args, variant).resolve()),
                               "--data", str(args.data.resolve()), "--model", model,
                               "--seed", str(seed), "--epochs", str(spec["epochs"]),
                               "--protocol", plan["protocol"], "--output", str(target.resolve())]
                    emit({"event": "start", "job": name, "device": device, "command": command})
                    with (args.output / (name + ".log")).open("w") as log:
                        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        stop.set()
                        raise RuntimeError("Training failed; inspect " + name + ".log")
                    report = json.loads(report_path.read_text())
                    verify_report(report, spec, seed, data_hash, driver_hash, plan["protocol"], model)
                    emit({"event": "completed", "job": name, "steps": report["training_steps"],
                          "metrics": report["final_metrics"], "seconds": report["elapsed_seconds"]})
                left = args.output / job_name(model, seed, "baseline", repeat)
                right = args.output / job_name(model, seed, "routed", repeat)
                comparison = args.output / ("{}-seed{}-repeat{}-comparison.json".format(model, seed, repeat))
                command = [sys.executable, str(driver), "--compare", str(left), str(right),
                           "--output", str(comparison)]
                for key, value in plan["strict_numeric_tolerances"].items():
                    command += ["--" + key.replace("_", "-"), str(value)]
                with comparison.with_suffix(".log").open("w") as log:
                    result = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                                            stdout=log, stderr=subprocess.STDOUT)
                if result.returncode not in (0, 1) or not comparison.exists():
                    stop.set()
                    raise RuntimeError("Comparison command failed")
                if not json.loads(comparison.read_text())["preconditions_equal"]:
                    stop.set()
                    raise RuntimeError("Unfair pairing: " + str(comparison))

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(worker, index, device) for index, device in enumerate(devices)]
        for future in futures:
            future.result()
    emit({"event": "selected_models_completed", "models": selected})


def summarize(args):
    import scipy

    plan = load_plan(args.plan)
    result = {"scope": plan["scope"], "plan_sha256": sha256(args.plan),
              "equivalence_rule": plan["equivalence"], "models": {},
              "statistics_driver_sha256": sha256(__file__),
              "statistics_environment": {"python": platform.python_version(), "scipy": scipy.__version__}}
    for model in args.models or plan["models"]:
        spec = plan["models"][model]
        rows, controls, cohorts = [], [], {}
        for seed in spec["seeds"]:
            paths = [args.output / job_name(model, seed, variant) / "report.json"
                     for variant in ("baseline", "routed")]
            reports = [json.loads(path.read_text()) for path in paths]
            for variant, report in zip(("baseline", "routed"), reports):
                verify_report(report, spec, seed, reports[0]["data_manifest_sha256"],
                              reports[0]["driver_sha256"], plan["protocol"], model)
                verify_cohort(cohorts, variant, report)
            comparison_path = args.output / ("{}-seed{}-repeat0-comparison.json".format(model, seed))
            comparison = json.loads(comparison_path.read_text())
            if not comparison["preconditions_equal"]:
                raise ValueError("Unfair paired run")
            if comparison["baseline_report_sha256"] != sha256(paths[0]) or comparison["routed_report_sha256"] != sha256(paths[1]):
                raise ValueError("Comparison report no longer matches raw runs")
            metrics = [r["final_metrics"][spec["primary_metric"]] * spec["metric_to_percentage_points"] for r in reports]
            rows.append({"seed": seed, "baseline": metrics[0], "routed": metrics[1],
                         "delta_percentage_points": metrics[1] - metrics[0],
                         "strict_numeric_passed": comparison["passed"],
                         "bitwise_training_state_equal": all(reports[0][key] == reports[1][key]
                             for key in ("final_state_sha256", "losses", "final_metrics")),
                         "steps_per_run": reports[0]["training_steps"],
                         "report_hashes": [sha256(path) for path in paths]})
            if seed in spec["repeat_seeds"]:
                for variant, original in zip(("baseline", "routed"), reports):
                    path = args.output / job_name(model, seed, variant, 1) / "report.json"
                    repeated = json.loads(path.read_text())
                    verify_report(repeated, spec, seed, original["data_manifest_sha256"],
                                  original["driver_sha256"], plan["protocol"], model)
                    verify_cohort(cohorts, variant, repeated)
                    for key in ("initial_state_sha256", "effective_config_sha256", "initial_rng_sha256",
                                "first_batch_sha256", "first_forward_rng_sha256", "learning_rates", "environment"):
                        if original[key] != repeated[key]:
                            raise ValueError("Unfair same-version repeat: " + key)
                    controls.append({"seed": seed, "variant": variant,
                                     "first": original["final_metrics"][spec["primary_metric"]] * spec["metric_to_percentage_points"],
                                     "repeat": repeated["final_metrics"][spec["primary_metric"]] * spec["metric_to_percentage_points"],
                                     "delta_percentage_points": (repeated["final_metrics"][spec["primary_metric"]] -
                                                                 original["final_metrics"][spec["primary_metric"]]) * spec["metric_to_percentage_points"],
                                     "report_sha256": sha256(path)})
        rule = plan["equivalence"]
        test = equivalent([row["delta_percentage_points"] for row in rows],
                          rule["margin_percentage_points"], rule["alpha"])
        result["models"][model] = {"provenance": spec["provenance"], "epochs": spec["epochs"],
                                   "metric": spec["primary_metric"], "pairs": rows, "same_version_controls": controls,
                                   "baseline_mean": statistics.mean(row["baseline"] for row in rows),
                                   "baseline_std": statistics.stdev(row["baseline"] for row in rows),
                                   "routed_mean": statistics.mean(row["routed"] for row in rows),
                                   "routed_std": statistics.stdev(row["routed"] for row in rows),
                                   "all_strict_numeric_passed": all(row["strict_numeric_passed"] for row in rows),
                                   "all_bitwise_training_state_equal": all(row["bitwise_training_state_equal"] for row in rows),
                                   "equivalence_test": test}
    result["status"] = "completed_for_declared_models"
    path = args.output / "multiseed-summary.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True))
    emit(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("resnet18", "faster_rcnn", "detr", "yolov5s", "vit_b16"))
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--routed", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--compat", type=Path)
    parser.add_argument("--devices", default="2")
    args = parser.parse_args()
    if args.summarize:
        summarize(args)
    else:
        if not all((args.baseline, args.routed, args.data)):
            parser.error("--baseline, --routed and --data are required for training")
        run_suite(args)


if __name__ == "__main__":
    main()
