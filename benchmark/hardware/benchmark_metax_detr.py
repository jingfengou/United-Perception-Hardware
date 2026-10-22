"""Single-seed native/v2 DETR accuracy, timing and allocator-memory comparison.

Uses the existing full PennFudan recipe and UP Runner without modifying either.
No per-step tensor copies, gradient hashes, file writes or device synchronization.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from validate_platform_equivalence import file_hash, source_manifest, write_json

PROFILES = ("native", "metax_deterministic_v2")
ORDER = ("native-0", "v2-0", "v2-1", "native-1")
STEPS, WARMUP, SEED, EPOCHS = 720, 60, 131, 12


def cache_manifest(directory):
    return {p.relative_to(directory).as_posix(): file_hash(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def clock():
    return time.perf_counter()


def distribution(values):
    if not values or not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError("Timing samples must be positive and finite")
    ordered = sorted(values)
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "p95": ordered[math.ceil(.95 * len(values)) - 1],
            "sum": sum(values)}


class TrainingMeter:
    def __init__(self, cuda):
        self.cuda = cuda
        self.started = False
        self.finished = False
        self.excluded = {"evaluation": 0., "checkpoint": 0.}
        self.peak_allocated = self.peak_reserved = 0
        self.events, self.rows = [], []

    def peak(self):
        self.peak_allocated = max(self.peak_allocated, self.cuda.max_memory_allocated())
        self.peak_reserved = max(self.peak_reserved, self.cuda.max_memory_reserved())

    def elapsed(self):
        return clock() - self.start - sum(self.excluded.values())

    def begin(self):
        self.cuda.synchronize()
        self.cuda.reset_peak_memory_stats()
        self.start = clock()
        self.started = True

    def before_forward(self, batch):
        self.pending = self.cuda.Event(enable_timing=True)
        self.pending.record()
        self.batch_size = int(batch["image"].shape[0])

    def after_update(self, loss, learning_rates):
        end = self.cuda.Event(enable_timing=True)
        end.record()
        self.events.append((self.pending, end))
        self.rows.append({"loss": loss.detach(), "images": self.batch_size, "lr": learning_rates})
        self.peak()
        if len(self.rows) == WARMUP:
            self.cuda.synchronize()
            self.warmup_seconds = self.elapsed()

    def outside_training(self, name, function, *args, **kwargs):
        if not self.started or self.finished:
            return function(*args, **kwargs)
        self.cuda.synchronize()
        self.peak()
        started = clock()
        try:
            return function(*args, **kwargs)
        finally:
            self.cuda.synchronize()
            self.excluded[name] += clock() - started
            self.cuda.reset_peak_memory_stats()

    def finish(self):
        self.cuda.synchronize()
        self.peak()
        self.training_seconds = self.elapsed()
        self.finished = True

    def report(self):
        if not self.finished or len(self.rows) != STEPS:
            raise ValueError("The complete 720-update schedule must finish")
        durations = [start.elapsed_time(end) for start, end in self.events]
        steady_seconds = self.training_seconds - self.warmup_seconds
        if steady_seconds <= 0:
            raise ValueError("Invalid steady training duration")
        return {"training_seconds_excluding_eval_save": self.training_seconds,
                "warmup_steps": WARMUP, "warmup_training_seconds": self.warmup_seconds,
                "steady_steps": STEPS - WARMUP, "steady_training_seconds": steady_seconds,
                "steady_wall_ms_per_update": 1000 * steady_seconds / (STEPS - WARMUP),
                "steady_images_per_second": sum(row["images"] for row in self.rows[WARMUP:]) / steady_seconds,
                "core_stream_ms": distribution(durations[WARMUP:]),
                "excluded_seconds": self.excluded,
                "training_peak_allocated_bytes": self.peak_allocated,
                "training_peak_reserved_bytes": self.peak_reserved,
                "per_step_core_stream_ms": durations}


def run(args):
    import torch
    from diagnose_detr_origin import cpu, digest, rng, setup
    from validate_training_accuracy import batch_hash, effective_config, prepare_legacy_environment

    args.output.mkdir(parents=True, exist_ok=False)
    prepared_at = clock()
    root = args.source.resolve()
    before_source = source_manifest(root)
    data_sha256 = file_hash(args.data)
    data = json.loads(args.data.read_text())
    prepare_legacy_environment()
    sys.path[:0] = [str(root / "packages/up-platform/src"), str(root)]
    import up
    from up.accelerator import configure_accelerator, get_accelerator
    configure_accelerator({"runtime": {"accelerator": {"backend": "metax", "numerics": args.profile}}})
    up.initialize()
    setup()
    from up.runner.base_runner import BaseRunner
    from up.utils.general.global_flag import DIST_BACKEND
    from up.utils.general.hook_helper import Hook
    DIST_BACKEND.backend = "dist"
    config, _ = effective_config(root, data, "detr", SEED, EPOCHS, "multiseed_v2")
    config["runtime"]["accelerator"] = {"backend": "metax", "numerics": args.profile}
    write_json(args.output / "effective_config.json", config)
    algorithm = copy.deepcopy(config)
    algorithm["runtime"]["accelerator"].pop("numerics")
    algorithm_sha256 = hashlib.sha256(json.dumps(algorithm, sort_keys=True).encode()).hexdigest()
    config["saver"].update(save_dir=str(args.output / "checkpoints"), results_dir=str(args.output / "predictions"))
    runner = BaseRunner(config, work_dir=str(args.output), training=True)
    assert runner.max_iter == STEPS and runner.train_epoch_size == WARMUP
    initial_model = digest(runner.model.state_dict())
    initial_optimizer = digest(runner.optimizer.state_dict())
    initial_rng = digest(rng())
    meter = TrainingMeter(torch.cuda)
    first_batch = []
    evaluations = []
    original_evaluate, original_save = runner.evaluate, runner.save

    def evaluate():
        def invoke():
            result = original_evaluate()
            evaluations.append({"after_updates": len(meter.rows), "metrics": dict(result)})
            assert get_accelerator().purpose == "training"
            return result
        return meter.outside_training("evaluation", invoke)

    runner.evaluate = evaluate
    runner.save = lambda *a, **kw: meter.outside_training("checkpoint", original_save, *a, **kw)

    class Recorder(Hook):
        def before_train(self):
            meter.begin()

        def before_forward(self, cur_iter, batch):
            if not first_batch:
                first_batch.append(batch_hash(batch))
            meter.before_forward(batch)

        def after_backward(self, cur_iter, loss):
            self.loss = loss

        def after_update(self, cur_iter):
            meter.after_update(self.loss, [float(group["lr"]) for group in runner.optimizer.param_groups])

        def after_train(self):
            meter.finish()

    runner._hooks.hooks.append(Recorder(runner))
    initial_metrics = dict(runner.evaluate())
    setup_seconds = clock() - prepared_at
    torch.cuda.synchronize()
    started = clock()
    runner.train()
    loop_seconds = clock() - started
    final_metrics = dict(runner.evaluate())
    losses = [float(row.pop("loss")) for row in meter.rows]
    assert all(math.isfinite(value) for value in losses)
    final = {"model": cpu(runner.model.state_dict()), "optimizer": cpu(runner.optimizer.state_dict()),
             "scheduler": cpu(runner.lr_scheduler.state_dict())}
    assert all(torch.isfinite(tensor).all() for tensor in final["model"].values())
    torch.save(final, args.output / "final.pt")
    write_json(args.output / "steps.json", [{**row, "loss": loss, "step": i + 1}
                                            for i, (row, loss) in enumerate(zip(meter.rows, losses))])
    timing = meter.report()
    assert source_manifest(root) == before_source and file_hash(args.data) == data_sha256
    report = {"status": "completed", "model": "detr", "profile": args.profile, "seed": SEED,
              "epochs": EPOCHS, "training_steps": STEPS, "source_manifest_sha256": before_source["sha256"],
              "data_sha256": data_sha256, "algorithm_config_sha256": algorithm_sha256,
              "initial_model_sha256": initial_model, "initial_optimizer_sha256": initial_optimizer,
              "initial_rng_sha256": initial_rng, "first_batch_sha256": first_batch[0],
              "final_model_sha256": digest(final["model"]), "final_optimizer_sha256": digest(final["optimizer"]),
              "final_metrics": final_metrics, "initial_metrics": initial_metrics, "evaluations": evaluations,
              "setup_and_initial_eval_seconds": setup_seconds, "train_loop_seconds": loop_seconds,
              "timing": timing, "steps_sha256": file_hash(args.output / "steps.json"),
              "checkpoint_sha256": file_hash(args.output / "final.pt"),
              "numerics": get_accelerator().numerics_state(), "driver_sha256": file_hash(Path(__file__)),
              "driver_normalized_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
              "environment": {"python": sys.version, "torch": torch.__version__,
                              "torchvision": __import__("torchvision").__version__,
                              "triton": __import__("importlib.metadata", fromlist=["version"]).version("triton"),
                              "gpu": torch.cuda.get_device_name(0),
                              "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                              "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                              "cudnn_benchmark": torch.backends.cudnn.benchmark,
                              "cudnn_deterministic": torch.backends.cudnn.deterministic,
                              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                              "matmul_precision": torch.get_float32_matmul_precision(),
                              "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR")}}
    write_json(args.output / "report.json", report)


def summarize(root):
    reports = {name: json.loads((root / name / "report.json").read_text()) for name in ORDER}
    reference = reports[ORDER[0]]
    invariants = ("seed", "epochs", "training_steps", "source_manifest_sha256", "data_sha256",
                  "algorithm_config_sha256", "initial_model_sha256", "initial_optimizer_sha256",
                  "initial_rng_sha256", "first_batch_sha256", "driver_sha256")
    for name, report in reports.items():
        assert report["status"] == "completed" and report["training_steps"] == STEPS
        assert report["profile"] == (PROFILES[0] if name.startswith("native") else PROFILES[1])
        assert all(reference[key] == report[key] for key in invariants), name
        assert report["timing"]["steady_steps"] == STEPS - WARMUP
        assert report["environment"]["deterministic_algorithms"] and not report["environment"]["deterministic_warn_only"]
        assert report["environment"]["visible_devices"] == reference["environment"]["visible_devices"]
        assert {k: v for k, v in report["environment"].items() if k != "triton_cache_dir"} == {
            k: v for k, v in reference["environment"].items() if k != "triton_cache_dir"}
        assert file_hash(root / name / "steps.json") == report["steps_sha256"]
        assert file_hash(root / name / "final.pt") == report["checkpoint_sha256"]
        assert report["timing"]["training_peak_reserved_bytes"] >= report["timing"]["training_peak_allocated_bytes"] > 0
        assert all(math.isfinite(float(value)) for value in report["final_metrics"].values())
        if name.startswith("native"):
            assert report["numerics"]["profile"] == "native" and not report["numerics"]["operations"]
        else:
            assert report["numerics"]["operations"]["aten.mm.default"] > 0
            assert report["numerics"]["operations"]["aten.convolution_backward.weight"] > 0
    steps = {name: json.loads((root / name / "steps.json").read_text()) for name in ORDER}
    assert all(len(rows) == STEPS for rows in steps.values())
    assert all([row["step"] for row in rows] == list(range(1, STEPS + 1))
               and all(math.isfinite(row["loss"]) for row in rows) for rows in steps.values())
    assert all([row["lr"] for row in rows] == [row["lr"] for row in steps[ORDER[0]]] for rows in steps.values())
    assert all([row["images"] for row in rows] == [row["images"] for row in steps[ORDER[0]]] for rows in steps.values())
    summary = {"scope": "same C500, one seed 131, ABBA order, two repeats per mode; not multi-seed or COCO convergence",
               "order": ORDER, "invariants": {key: reference[key] for key in invariants},
               "rtol": 1e-4, "atol": 1e-5, "metric_atol": 1e-6, "runs": reports, "modes": {}}
    for mode in ("native", "v2"):
        selected = [reports[mode + "-" + str(i)] for i in range(2)]
        summary["modes"][mode] = {"final_ap": [row["final_metrics"]["bbox.AP"] for row in selected],
                                  "final_weights_repeat_exact": selected[0]["final_model_sha256"] == selected[1]["final_model_sha256"],
                                  "losses_repeat_exact": [row["loss"] for row in steps[mode + "-0"]] == [row["loss"] for row in steps[mode + "-1"]]}
        for metric in ("steady_wall_ms_per_update", "steady_images_per_second", "training_peak_allocated_bytes",
                       "training_peak_reserved_bytes", "training_seconds_excluding_eval_save", "warmup_training_seconds"):
            values = [row["timing"][metric] for row in selected]
            assert all(math.isfinite(value) and value > 0 for value in values)
            summary["modes"][mode][metric] = {"values": values, "mean": statistics.mean(values)}
    native, fixed = summary["modes"]["native"], summary["modes"]["v2"]
    summary["comparison"] = {
        "steady_time_ratio_v2_over_native": fixed["steady_wall_ms_per_update"]["mean"] / native["steady_wall_ms_per_update"]["mean"],
        "throughput_ratio_v2_over_native": fixed["steady_images_per_second"]["mean"] / native["steady_images_per_second"]["mean"],
        "allocated_ratio_v2_over_native": fixed["training_peak_allocated_bytes"]["mean"] / native["training_peak_allocated_bytes"]["mean"],
        "reserved_ratio_v2_over_native": fixed["training_peak_reserved_bytes"]["mean"] / native["training_peak_reserved_bytes"]["mean"],
        "mean_ap_difference_v2_minus_native": statistics.mean(fixed["final_ap"]) - statistics.mean(native["final_ap"])}
    write_json(root / "summary.json", summary)
    return summary


def suite(args):
    import fcntl
    from prepare_cross_vendor_data import verify_data
    args.output.mkdir(parents=True, exist_ok=False)
    lock = (args.output.parent / ("metax-device-" + args.device + ".lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = source_manifest(args.source)
    data = file_hash(args.data)
    verified_data = verify_data(json.loads(args.data.read_text()))
    status = {"status": "running", "pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "source": source, "data_sha256": data, "verified_data": verified_data, "jobs": [], "device": args.device}
    if args.cache_from:
        args.cache_from = args.cache_from.resolve()
        status.update(cache=str(args.cache_from), cache_before=cache_manifest(args.cache_from))
        if not status["cache_before"]:
            raise ValueError("Warm-cache control requires a populated Triton cache")
    write_json(args.output / "execution.json", status)
    try:
        for name in ORDER:
            profile = PROFILES[0] if name.startswith("native") else PROFILES[1]
            cache = args.cache_from or args.output / (name + "-triton-cache")
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.device, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                               PYTHONHASHSEED="0", DEFAULT_TASKS="cls:det", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                               PYTORCH_DISABLE_CUDA_CUDNN_TF32="1", TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
                               TRITON_CACHE_DIR=str(cache), USER=os.environ.get("USER", "up-validation"),
                               LOGNAME=os.environ.get("LOGNAME", "up-validation"))
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "run", "--source", str(args.source),
                       "--data", str(args.data), "--output", str(args.output / name), "--profile", profile]
            status["current"] = name
            write_json(args.output / "execution.json", status)
            with (args.output / (name + ".log")).open("xb") as log:
                result = subprocess.run(command, cwd=args.source, env=environment, stdout=log, stderr=subprocess.STDOUT)
            status["jobs"].append({"name": name, "command": command, "exit_code": result.returncode})
            write_json(args.output / "execution.json", status)
            if result.returncode:
                raise RuntimeError(name + " failed; preserve this directory before any new attempt")
            assert source_manifest(args.source) == source and file_hash(args.data) == data
        summarize(args.output)
        if args.cache_from:
            status["cache_after"] = cache_manifest(args.cache_from)
            status["cache_unchanged"] = status["cache_after"] == status["cache_before"]
        status.update(status="completed", current=None, finished_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception:
        status.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        write_json(args.output / "execution.json", status)
        fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "suite", "summarize"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="2")
    parser.add_argument("--cache-from", type=Path, help="Reuse a completed v2 cohort's full Triton cache")
    parser.add_argument("--profile", choices=PROFILES, default="native")
    args = parser.parse_args()
    if args.action != "summarize" and (args.source is None or args.data is None):
        parser.error("run/suite require --source and --data")
    if args.action == "summarize":
        summarize(args.output)
    else:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        {"run": run, "suite": suite}[args.action](args)


if __name__ == "__main__":
    main()
