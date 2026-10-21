"""Run the real UP Runner in separate upstream/routed processes, then compare."""
import argparse
import collections
import collections.abc
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare_legacy_environment():
    import numpy as np

    collections.Iterable = collections.abc.Iterable
    np.float = float
    return ["collections.Iterable = collections.abc.Iterable (both processes)",
            "numpy.float = float (original legacy dtype alias, both processes)"]


def effective_config(root, data, model, seed, epochs):
    import yaml

    path = ("configs/cls/resnet/res18.yaml" if model == "resnet18" else
            "configs/det/faster_rcnn/faster_rcnn_r50_fpn_1x.yaml")
    config = yaml.safe_load((root / path).read_text())
    config.setdefault("runtime", {}).update(
        device="cuda", random_seed=seed, accelerator={"backend": "nvidia"})
    config["args"] = {"backend": "dist", "display": 50, "no_running_config": True}
    config["hooks"] = []
    config["trainer"].update(max_epoch=epochs, test_freq=1, save_freq=epochs,
                             only_save_latest=False)
    config["trainer"]["lr_scheduler"].update(warmup_iter=0)
    config["trainer"]["lr_scheduler"]["kwargs"]["milestones"] = [max(1, epochs - 1)]
    config["trainer"]["optimizer"]["kwargs"]["lr"] = .05 if model == "resnet18" else .0025
    config["saver"] = {"auto_resume": False, "save_result": True}
    selected = data["cifar10" if model == "resnet18" else "pennfudan"]
    for split in ("train", "test"):
        kwargs = config["dataset"][split]["dataset"]["kwargs"]
        kwargs["meta_file"] = selected[split]["meta"]
        kwargs["image_reader"]["kwargs"]["image_dir"] = selected[split]["image_dir"]
        if split == "test":
            evaluator = kwargs["evaluator"]["kwargs"]
            if model == "faster_rcnn":
                evaluator["gt_file"] = selected[split]["meta"]
        for transform in kwargs["transformer"]:
            options = transform.get("kwargs", {})
            if transform["type"] in ("torch_random_resized_crop", "torch_center_crop"):
                options["size"] = 64
            elif transform["type"] == "torch_resize":
                options["size"] = 72
            elif transform["type"] == "keep_ar_resize":
                options.update(scales=[320], max_size=512)
    if model == "resnet18":
        config["num_classes"] = 10
        config["net"][1]["kwargs"]["num_classes"] = 10
        config["net"][2]["kwargs"]["cls_loss"]["kwargs"]["num_classes"] = 10
        for split in ("train", "test"):
            config["dataset"][split]["batch_sampler"]["kwargs"]["batch_size"] = 64
            config["dataset"][split]["dataloader"]["kwargs"].update(num_workers=0, pin_memory=True)
    else:
        config["num_classes"] = 2
        for module in config["net"]:
            if "num_classes" in module.get("kwargs", {}):
                module["kwargs"]["num_classes"] = 2
        config["saver"]["pretrain_model"] = data["resnet50_weights"]["path"]
        config["dataset"]["dataloader"]["kwargs"].update(num_workers=0, pin_memory=True)
    return config, path


def run(args):
    import torch

    root = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to prevent accidental resume")
    output.mkdir(parents=True, exist_ok=True)
    data_path = args.data.resolve()
    data = json.loads(data_path.read_text())
    # The unmodified 2022 upstream imports this removed standard-library alias.
    # Both processes get the same import-only compatibility shim, no kernel shims.
    compatibility_shims = prepare_legacy_environment()
    os.environ["DEFAULT_TASKS"] = "cls:det"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, str(root))
    import up
    initialize = vars(up).get("initialize")
    if initialize is not None and getattr(initialize, "__module__", None) == "up":
        initialize()
    if Path(up.__file__).resolve().parent != root / "up":
        raise RuntimeError("Imported the wrong UP source tree")
    from up.utils.general.global_flag import DIST_BACKEND
    from up.utils.general.hook_helper import Hook
    from up.runner.base_runner import BaseRunner

    DIST_BACKEND.backend = "dist"
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0)
    config, original_config = effective_config(root, data, args.model, args.seed, args.epochs)
    config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    (output / "effective_config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    config["saver"].update(save_dir=str(output / "checkpoints"), results_dir=str(output / "predictions"))
    runner = BaseRunner(config, work_dir=str(output), training=True)

    class Recorder(Hook):
        def __init__(self, owner):
            super().__init__(owner)
            self.losses = []
            self.metrics = []

        def after_backward(self, cur_iter, loss):
            value = float(loss.detach())
            if not math.isfinite(value):
                raise ValueError("Non-finite training loss at {}".format(cur_iter))
            self.losses.append(value)

        def after_eval(self, metrics):
            self.metrics.append({str(k): float(v) for k, v in metrics.items()
                                 if isinstance(v, (float, int))})

    recorder = Recorder(runner)
    runner._hooks.hooks.append(recorder)
    initial_hash = state_hash(runner.model.state_dict())
    started = time.monotonic()
    initial_metrics = dict(runner.evaluate())
    runner.train()
    final_metrics = dict(runner.evaluate())
    final_state = {k: v.detach().cpu().clone() for k, v in runner.model.state_dict().items()}
    torch.save({"model": final_state, "optimizer": runner.optimizer.state_dict(),
                "scheduler": runner.lr_scheduler.state_dict()}, output / "comparison_state.pt")
    # Reload through a newly constructed original ModelHelper and optimizer.
    saved = torch.load(output / "comparison_state.pt", map_location="cpu", weights_only=False)
    restored = type(runner.model)(copy.deepcopy(config["net"])).to("cuda")
    restored.load_state_dict(copy.deepcopy(saved["model"]), strict=True)
    runner.model = restored
    runner.optimizer = runner.build_optimizer(copy.deepcopy(config["trainer"]["optimizer"]))
    runner.optimizer.load_state_dict(saved["optimizer"])
    restored_metrics = dict(runner.evaluate())
    if final_metrics != restored_metrics:
        raise AssertionError("Restored model changed validation metrics")
    sources = {str(path.relative_to(root)): file_hash(path) for path in sorted((root / "up").rglob("*.py"))}
    binaries = {str(path.relative_to(root)): file_hash(path) for path in sorted((root / "up").rglob("*.so"))}
    report = {
        "status": "completed", "scope": "real-data short-training regression, not full benchmark convergence",
        "model": args.model, "seed": args.seed, "epochs": args.epochs,
        "source_root": str(root), "original_config": original_config,
        "source_manifest_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        "source_files": sources, "native_binaries": binaries,
        "effective_config_sha256": config_digest, "data_manifest_sha256": file_hash(data_path),
        "data": data["cifar10" if args.model == "resnet18" else "pennfudan"],
        "initial_state_sha256": initial_hash, "final_state_sha256": state_hash(final_state),
        "training_steps": len(recorder.losses), "losses": recorder.losses,
        "initial_metrics": initial_metrics, "epoch_metrics": recorder.metrics,
        "final_metrics": final_metrics, "restored_metrics": restored_metrics,
        "checkpoint_restore": "new model and optimizer loaded; validation metrics identical",
        "elapsed_seconds": time.monotonic() - started,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "deterministic_algorithms": True, "tf32": False,
                        "default_tasks": os.environ["DEFAULT_TASKS"]},
        "compatibility_shims": compatibility_shims,
        "driver_sha256": file_hash(__file__),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: report[k] for k in ("status", "model", "seed", "training_steps", "final_metrics")}, indent=2))


def compare(args):
    import torch

    baseline_path, routed_path = args.compare
    baseline = json.loads((baseline_path / "report.json").read_text())
    routed = json.loads((routed_path / "report.json").read_text())
    differences = []
    for report in (baseline, routed):
        if report["status"] != "completed":
            differences.append("A training run did not complete")
        if report["training_steps"] <= 0 or len(report["losses"]) != report["training_steps"]:
            differences.append("A training run has no valid complete loss trace")
    for key in ("status", "model", "seed", "epochs", "training_steps", "effective_config_sha256",
                "data_manifest_sha256", "initial_state_sha256", "driver_sha256", "environment"):
        if baseline[key] != routed[key]:
            differences.append("Mismatched " + key)
    states = [torch.load(path / "comparison_state.pt", map_location="cpu", weights_only=False)
              for path in args.compare]
    weights = {"max_abs_difference": 0., "not_close": []}
    if states[0]["model"].keys() != states[1]["model"].keys():
        differences.append("Different checkpoint keys")
    else:
        for key, value in states[0]["model"].items():
            other = states[1]["model"][key]
            if value.shape != other.shape or value.dtype != other.dtype:
                differences.append("Different checkpoint tensor contract: " + key)
                continue
            delta = float((value.double() - other.double()).abs().max()) if value.numel() else 0.
            weights["max_abs_difference"] = max(weights["max_abs_difference"], delta)
            if not torch.allclose(value, other, rtol=args.rtol, atol=args.atol):
                weights["not_close"].append(key)
    if weights["not_close"]:
        differences.append("Final weights outside numerical tolerance")
    a, b = torch.tensor(baseline["losses"], dtype=torch.float64), torch.tensor(routed["losses"], dtype=torch.float64)
    loss_close = a.shape == b.shape and torch.allclose(a, b, rtol=args.rtol, atol=args.atol)
    if not loss_close:
        differences.append("Training losses outside numerical tolerance")
    metric_deltas = {}
    if baseline["final_metrics"].keys() != routed["final_metrics"].keys():
        differences.append("Different metric keys")
    else:
        for key, value in baseline["final_metrics"].items():
            if isinstance(value, (int, float)):
                delta = float(routed["final_metrics"][key]) - float(value)
                metric_deltas[key] = delta
                if not math.isfinite(delta) or abs(delta) > args.metric_atol:
                    differences.append("Metric outside tolerance: " + key)
    report = {"passed": not differences, "differences": differences,
              "scope": baseline["scope"], "model": baseline["model"], "seed": baseline["seed"],
              "epochs": baseline["epochs"], "training_steps_per_version": baseline["training_steps"],
              "baseline_metrics": baseline["final_metrics"], "routed_metrics": routed["final_metrics"],
              "metric_deltas": metric_deltas, "weights": weights,
              "losses_close": bool(loss_close),
              "max_loss_abs_difference": float((a - b).abs().max()) if a.shape == b.shape and a.numel() else None,
              "tolerances": {"atol": args.atol, "rtol": args.rtol, "metric_atol": args.metric_atol},
              "baseline_report_sha256": file_hash(baseline_path / "report.json"),
              "routed_report_sha256": file_hash(routed_path / "report.json")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model", choices=("resnet18", "faster_rcnn"))
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, nargs=2)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--metric-atol", type=float, default=1e-6)
    args = parser.parse_args()
    if args.compare:
        return compare(args)
    if not all((args.source, args.data, args.model)):
        parser.error("--source, --data and --model are required for a training run")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
