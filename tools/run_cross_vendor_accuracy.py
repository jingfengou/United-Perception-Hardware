"""Execute and audit fixed real-data UP training cohorts across vendor stacks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUITE = sibling("run_multiseed_accuracy")
DATA = sibling("prepare_cross_vendor_data")
sha256 = DATA.sha256


def load_plan(path):
    plan = json.loads(path.read_text())
    if plan["backends"] != ["nvidia", "metax"]:
        raise ValueError("This protocol requires both declared vendor stacks")
    for spec in plan["models"].values():
        if len(spec["seeds"]) < 2 or len(set(spec["seeds"])) != len(spec["seeds"]):
            raise ValueError("Need distinct predeclared seeds")
        if spec["protocol"] not in ("multiseed_v1", "multiseed_v2") or spec["epochs"] <= 0:
            raise ValueError("Invalid model schedule")
        if spec["expected_steps_per_run"] <= 0:
            raise ValueError("Missing complete step budget")
    return plan


def verify_run(report, plan, model, seed, backend, data_hash, driver_hash):
    spec = plan["models"][model]
    SUITE.verify_report(report, spec, seed, data_hash, driver_hash, spec["protocol"], model)
    if report["backend"] != backend:
        raise ValueError("Wrong vendor identity")
    environment = report["environment"]
    if environment["tf32"] or environment["matmul_allow_tf32"] or environment["cudnn_allow_tf32"]:
        raise ValueError("The predeclared strict FP32 policy was not applied")
    if environment["matmul_precision"] != "highest" or not environment["deterministic_algorithms"]:
        raise ValueError("Changed precision/determinism policy")
    if backend == "metax" and report["nvidia_extension_loaded"]:
        raise ValueError("NVIDIA binary was loaded on the MetaX path")
    if plan.get("disable_convolution_library", False) and report["environment"].get("convolution_library_enabled") is not False:
        raise ValueError("The declared convolution-library disable policy was not applied")
    if plan.get("audit_all_training_batches", False):
        audit = report.get("batch_audit", [])
        if len(audit) != spec["expected_steps_per_run"] or any(
                row["step"] != index for index, row in enumerate(audit)):
            raise ValueError("Incomplete or unordered training batch audit")
        if audit[0]["batch_sha256"] != report["first_batch_sha256"]:
            raise ValueError("First training batch disagrees with complete batch audit")
        digest = hashlib.sha256(json.dumps([row["batch_sha256"] for row in audit]).encode()).hexdigest()
        if report.get("all_training_batches_sha256") != digest:
            raise ValueError("Training batch audit hash mismatch")


def compare_pair(nvidia, metax, plan):
    if nvidia["backend"] != "nvidia" or metax["backend"] != "metax":
        raise ValueError("Pair requires different, explicitly named vendors")
    for key in ("model", "seed", "epochs", "protocol", "training_steps", "expected_training_steps", *plan["pairing"]):
        if key not in nvidia or key not in metax or nvidia[key] != metax[key]:
            raise ValueError("Unfair cross-vendor pair: " + key)
    return {"preconditions_equal": True,
            "environment_equal": nvidia["environment"] == metax["environment"],
            "gpu_rng_equal_diagnostic_only": nvidia["initial_rng_sha256"] == metax["initial_rng_sha256"],
            "bitwise_final_state_equal": nvidia["final_state_sha256"] == metax["final_state_sha256"],
            "losses_equal": nvidia["losses"] == metax["losses"],
            "max_loss_absolute_difference": max(abs(a - b) for a, b in zip(nvidia["losses"], metax["losses"]))}


def run_suite(args):
    plan = load_plan(args.plan)
    integrity = DATA.verify_data(json.loads(args.data.read_text()))
    driver = Path(__file__).with_name("validate_training_accuracy.py")
    selected = args.models or list(plan["models"])
    devices = args.devices.split(",")
    if any(not device.isdecimal() for device in devices) or len(devices) != len(set(devices)):
        raise ValueError("Each worker requires a unique numeric physical device")
    args.output.mkdir(parents=True, exist_ok=True)
    execution = {"backend": args.backend, "plan_sha256": sha256(args.plan),
                 "data_sha256": sha256(args.data), "driver_sha256": sha256(driver),
                 "suite_sha256": sha256(__file__), "data_tool_sha256": sha256(DATA.__file__),
                 "statistics_tool_sha256": sha256(SUITE.__file__),
                 "source": str(args.source.resolve()), "data": str(args.data.resolve()),
                 "models": selected, "devices": devices, "verified_data": integrity}
    execution["source_files"] = {str(path.relative_to(args.source)): sha256(path)
                                 for folder in ("up", "configs") for path in sorted((args.source / folder).rglob("*"))
                                 if path.is_file() and path.suffix in (".py", ".yaml", ".json", ".so")}
    execution["adapter_sha256"] = sha256(driver.with_name("accuracy_detr_adapter.py"))
    manifest = args.output / ("execution-" + "-".join(selected) + ".json")
    if manifest.exists() and json.loads(manifest.read_text()) != execution:
        raise ValueError("Changed experiment; use a new output directory")
    manifest.write_text(json.dumps(execution, indent=2, sort_keys=True))
    jobs = queue.Queue()
    # Long CPU-augmentation jobs are queued first, without altering model seeds or budgets.
    for model in sorted(selected, key=lambda name: name != "yolov5s"):
        for seed in plan["models"][model]["seeds"]:
            jobs.put((model, seed))

    def worker(device):
        while True:
            try:
                model, seed = jobs.get_nowait()
            except queue.Empty:
                return
            spec = plan["models"][model]
            output = args.output / (model + "-seed" + str(seed))
            report_path = output / "report.json"
            if report_path.exists():
                verify_run(json.loads(report_path.read_text()), plan, model, seed, args.backend,
                           execution["data_sha256"], execution["driver_sha256"])
                SUITE.emit({"reused_verified_complete_run": str(output)})
                continue
            if output.exists() and any(output.iterdir()):
                raise ValueError("Preserve partial run; use a new attempt directory: " + str(output))
            if args.backend == "nvidia":
                SUITE.ensure_device_idle(device)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=device, OMP_NUM_THREADS="2",
                               MKL_NUM_THREADS="2", PYTHONHASHSEED="0", DEFAULT_TASKS="cls:det",
                               PYTORCH_DISABLE_CUDA_CUDNN_TF32="1", TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0")
            command = [sys.executable, str(driver), "--source", str(args.source.resolve()),
                       "--data", str(args.data.resolve()), "--backend", args.backend,
                       "--model", model, "--seed", str(seed), "--epochs", str(spec["epochs"]),
                       "--protocol", spec["protocol"], "--output", str(output.resolve())]
            if plan.get("audit_all_training_batches", False):
                command.append("--audit-batches")
            if plan.get("disable_convolution_library", False):
                command.append("--disable-convolution-library")
            SUITE.emit({"starting": model, "seed": seed, "backend": args.backend, "device": device})
            with (args.output / (output.name + ".log")).open("w") as log:
                result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError("Training failed; partial evidence retained: " + str(output))
            verify_run(json.loads(report_path.read_text()), plan, model, seed, args.backend,
                       execution["data_sha256"], execution["driver_sha256"])
            SUITE.emit({"completed": model, "seed": seed, "backend": args.backend})

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(worker, device) for device in devices]
        for future in futures:
            future.result()
    SUITE.emit({"status": "completed_vendor_cohort", "backend": args.backend})


def summarize(args):
    plan = load_plan(args.plan)
    result = {"status": "completed_for_declared_models", "plan_sha256": sha256(args.plan),
              "scope": plan["scope"], "software_scope": plan["software_scope"],
              "delta_direction": "MetaX minus NVIDIA, percentage points", "models": {},
              "statistics_driver_sha256": sha256(__file__)}
    expected_data = expected_driver = None
    for model, spec in plan["models"].items():
        pairs, cohorts = [], {}
        for seed in spec["seeds"]:
            paths = [root / (model + "-seed" + str(seed)) / "report.json"
                     for root in (args.nvidia, args.metax)]
            reports = [json.loads(path.read_text()) for path in paths]
            if expected_data is None:
                expected_data, expected_driver = reports[0]["data_manifest_sha256"], reports[0]["driver_sha256"]
            for backend, report in zip(plan["backends"], reports):
                verify_run(report, plan, model, seed, backend, expected_data, expected_driver)
                SUITE.verify_cohort(cohorts, backend, report)
            pair = compare_pair(*reports, plan)
            metrics = [r["final_metrics"][spec["primary_metric"]] * spec["metric_to_percentage_points"] for r in reports]
            pair.update(seed=seed, nvidia=metrics[0], metax=metrics[1],
                        delta_percentage_points=metrics[1] - metrics[0],
                        steps_per_run=reports[0]["training_steps"],
                        report_sha256=[sha256(path) for path in paths])
            pairs.append(pair)
        test = SUITE.equivalent([pair["delta_percentage_points"] for pair in pairs],
                                plan["equivalence"]["margin_percentage_points"], plan["equivalence"]["alpha"])
        result["models"][model] = {"metric": spec["primary_metric"], "provenance": spec["provenance"],
                                   "epochs": spec["epochs"], "pairs": pairs, "equivalence_test": test,
                                   "nvidia_mean": statistics.mean(pair["nvidia"] for pair in pairs),
                                   "nvidia_std": statistics.stdev(pair["nvidia"] for pair in pairs),
                                   "metax_mean": statistics.mean(pair["metax"] for pair in pairs),
                                   "metax_std": statistics.stdev(pair["metax"] for pair in pairs),
                                   "environments": {backend: cohort["environment"] for backend, cohort in cohorts.items()}}
    result["all_models_equivalent"] = all(value["equivalence_test"]["equivalent_within_margin"]
                                           for value in result["models"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    SUITE.emit(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--nvidia", type=Path)
    parser.add_argument("--metax", type=Path)
    parser.add_argument("--backend", choices=("nvidia", "metax"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--models", nargs="+", choices=("resnet18", "yolov5s", "detr", "vit_b16"))
    args = parser.parse_args()
    if args.summarize:
        if not all((args.nvidia, args.metax)):
            parser.error("--nvidia and --metax are required to summarize paired cohorts")
        summarize(args)
    else:
        if not all((args.backend, args.source, args.data)):
            parser.error("--backend, --source and --data are required for training")
        run_suite(args)


if __name__ == "__main__":
    main()
