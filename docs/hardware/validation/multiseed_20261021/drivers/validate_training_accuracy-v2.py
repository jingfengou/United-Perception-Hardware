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
    np.int = int
    return ["collections.Iterable = collections.abc.Iterable (both processes)",
            "numpy.float = float (original legacy dtype alias, both processes)",
            "numpy.int = int (original AutoAugment dtype alias, both processes)"]


def effective_config(root, data, model, seed, epochs, protocol="legacy"):
    import yaml

    if model in ("yolov5s", "vit_b16"):
        if protocol != "multiseed_v2":
            raise ValueError("Additional models require the predeclared multiseed_v2 protocol")
        return additional_model_config(root, data, model, seed, epochs)
    if model == "detr" and protocol == "multiseed_v2":
        config, path = effective_config(root, data, model, seed, epochs, "multiseed_v1")
        config["trainer"]["optimizer"]["special_param_group"] = [
            {"key": "detector.backbone", "lr": 1e-5}]
        config["trainer"]["lr_scheduler"]["warmup_register_type"] = "no_scale_lr"
        return config, path
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
            if model != "resnet18":
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
    if protocol == "multiseed_v1":
        ratios = (.6, .9) if model == "resnet18" else (2 / 3, 11 / 12)
        config["trainer"]["lr_scheduler"]["kwargs"]["milestones"] = sorted({
            max(1, int(epochs * ratio)) for ratio in ratios})
        if model == "faster_rcnn":
            config["trainer"]["lr_scheduler"]["warmup_iter"] = 60
    elif protocol != "legacy":
        raise ValueError("Unknown accuracy protocol: " + protocol)
    if model == "detr":
        if protocol != "multiseed_v1":
            raise ValueError("DETR requires the explicitly labelled multiseed_v1 adapter protocol")
        config["saver"].pop("pretrain_model", None)
        config["net"] = [{"name": "detr", "type": "accuracy_detr_adapter.DetrForUP",
                          "kwargs": {"source": data["detr"]["source"],
                                     "checkpoint": data["detr"]["checkpoint"],
                                     "source_sha256": data["detr"]["source_sha256"],
                                     "checkpoint_sha256": data["detr"]["checkpoint_sha256"]}}]
        config["trainer"]["optimizer"] = {
            "type": "AdamW", "kwargs": {"lr": 1e-4, "weight_decay": 1e-4}}
        config["hooks"] = [{"type": "grad_clipper", "kwargs": {
            "mode": "pre_defined", "max_norm": .1, "norm_type": 2}}]
    return config, path


def additional_model_config(root, data, model, seed, epochs):
    import yaml

    classification = model == "vit_b16"
    path = ("configs/cls/vision_transformer/vit_b16_224.yaml" if classification else
            "configs/det/yolov5/yolov5_small_silu.yaml")
    config = yaml.safe_load((root / path).read_text())
    config.setdefault("runtime", {}).update(
        device="cuda", random_seed=seed, accelerator={"backend": "nvidia"})
    config["args"] = {"backend": "dist", "display": 50, "no_running_config": True}
    config["hooks"] = []
    config["trainer"].update(max_epoch=epochs, test_freq=1 if classification else 10,
                             save_freq=epochs, only_save_latest=False)
    config["saver"] = {"auto_resume": False, "save_result": True}
    selected = data["cifar10" if classification else "pennfudan"]
    config["num_classes"] = 10 if classification else 2
    for module in config["net"]:
        if "num_classes" in module.get("kwargs", {}):
            module["kwargs"]["num_classes"] = config["num_classes"]
    for split in ("train", "test"):
        dataset = config["dataset"][split]
        kwargs = dataset["dataset"]["kwargs"]
        kwargs["meta_file"] = selected[split]["meta"]
        kwargs["image_reader"]["kwargs"]["image_dir"] = selected[split]["image_dir"]
        if not classification and split == "test":
            kwargs["evaluator"]["kwargs"]["gt_file"] = selected[split]["meta"]
        for transform in kwargs["transformer"]:
            options = transform.setdefault("kwargs", {})
            if classification and transform["type"] in (
                    "torch_random_resized_crop", "torch_center_crop", "torch_auto_augmentation"):
                options["size"] = 64
            elif classification and transform["type"] == "torch_resize":
                options["size"] = 72
            elif transform["type"] == "mosaic":
                options["tar_size"] = 320
            elif transform["type"] == "random_perspective":
                options["border"] = [-160, -160]
            elif transform["type"] == "keep_ar_resize_max":
                options["max_size"] = 320
        if classification:
            dataset["batch_sampler"]["kwargs"]["batch_size"] = 64
            dataset["dataloader"]["kwargs"].update(num_workers=0, pin_memory=True)
            if split == "train":
                dataset["dataloader"]["kwargs"]["batch_fn"]["kwargs"]["num_classes"] = 10
    if classification:
        # Keep all 12 blocks and 768-wide features; only the CIFAR input/token grid changes.
        config["net"][0]["kwargs"]["image_size"] = 64
        config["net"][2]["kwargs"]["cls_loss"]["kwargs"]["num_classes"] = 10
        config["trainer"]["lr_scheduler"].update(warmup_iter=157, warmup_ratio=.01)
        config["trainer"]["lr_scheduler"]["kwargs"].update(T_max=epochs, warm_epoch=1)
    else:
        config["ema"]["enable"] = False
        config["dataset"]["batch_sampler"]["kwargs"]["batch_size"] = 64
        config["dataset"]["dataloader"]["kwargs"].update(num_workers=0, pin_memory=True)
        anchor = data["yolov5_anchors"]["path"]
        config["net"][-1]["kwargs"]["cfg"]["anchor_generator"]["kwargs"]["base_anchors_file"] = anchor
        scheduler = config["trainer"]["lr_scheduler"]
        scheduler.update(warmup_iter=0, warmup_epochs=3, warmup_total_epochs=epochs)
        scheduler["kwargs"]["T_max"] = epochs
    return config, path


def rng_hash():
    import pickle
    import random
    import numpy as np
    import torch

    state = [random.getstate(), np.random.get_state(), torch.get_rng_state().numpy(),
             torch.cuda.get_rng_state().cpu().numpy()]
    return hashlib.sha256(pickle.dumps(state)).hexdigest()


def batch_hash(value):
    import numpy as np
    import torch

    def describe(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous().numpy()
        if isinstance(item, np.ndarray):
            return [str(item.dtype), list(item.shape), hashlib.sha256(item.tobytes()).hexdigest()]
        if isinstance(item, dict):
            return {str(key): describe(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [describe(child) for child in item]
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError("Unrecorded batch value: " + str(type(item)))
    return hashlib.sha256(json.dumps(describe(value), sort_keys=True).encode()).hexdigest()


def run(args):
    import torch

    root = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to prevent accidental resume")
    output.mkdir(parents=True, exist_ok=True)
    data_path = args.data.resolve()
    data = json.loads(data_path.read_text())
    if args.model == "yolov5s" and file_hash(data["yolov5_anchors"]["path"]) != data["yolov5_anchors"]["sha256"]:
        raise ValueError("Pinned YOLO anchor file changed")
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
    config, original_config = effective_config(root, data, args.model, args.seed, args.epochs, args.protocol)
    config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    (output / "effective_config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    config["saver"].update(save_dir=str(output / "checkpoints"), results_dir=str(output / "predictions"))
    runner = BaseRunner(config, work_dir=str(output), training=True)

    class Recorder(Hook):
        def __init__(self, owner):
            super().__init__(owner)
            self.losses = []
            self.metrics = []
            self.learning_rates = []
            self.first_batch_sha256 = None
            self.first_forward_rng_sha256 = None

        def before_forward(self, cur_iter, batch):
            self.learning_rates.append([float(group["lr"]) for group in self.runner_ref().optimizer.param_groups])
            if self.first_batch_sha256 is None:
                self.first_batch_sha256 = batch_hash(batch)
                self.first_forward_rng_sha256 = rng_hash()

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
    initial_rng = rng_hash()
    started = time.monotonic()
    initial_metrics = dict(runner.evaluate())
    runner.train()
    if len(recorder.losses) != runner.max_iter:
        raise AssertionError("The entire declared optimizer-step budget was not completed")
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
        "status": "completed", "scope": "complete declared real-data regression schedule, not full ImageNet/COCO convergence",
        "model": args.model, "seed": args.seed, "epochs": args.epochs,
        "protocol": args.protocol,
        "model_provenance": "external official DETR adapter in both UP trees" if args.model == "detr" else "upstream UP model",
        "source_root": str(root), "original_config": original_config,
        "source_manifest_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        "source_files": sources, "native_binaries": binaries,
        "effective_config_sha256": config_digest, "data_manifest_sha256": file_hash(data_path),
        "data": data["cifar10" if args.model in ("resnet18", "vit_b16") else "pennfudan"],
        "initial_state_sha256": initial_hash, "final_state_sha256": state_hash(final_state),
        "initial_rng_sha256": initial_rng,
        "first_batch_sha256": recorder.first_batch_sha256,
        "first_forward_rng_sha256": recorder.first_forward_rng_sha256,
        "learning_rates": recorder.learning_rates,
        "expected_training_steps": runner.max_iter,
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
        "adapter_sha256": file_hash(Path(__file__).with_name("accuracy_detr_adapter.py")) if args.model == "detr" else None,
        "external_model": data.get("detr") if args.model == "detr" else None,
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
    for key in ("protocol", "initial_rng_sha256", "first_batch_sha256", "first_forward_rng_sha256",
                "learning_rates", "expected_training_steps", "adapter_sha256", "external_model"):
        if baseline.get(key) != routed.get(key):
            differences.append("Mismatched " + key)
    precondition_differences = list(differences)
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
              "preconditions_equal": not precondition_differences,
              "precondition_differences": precondition_differences,
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
    parser.add_argument("--model", choices=("resnet18", "faster_rcnn", "detr", "yolov5s", "vit_b16"))
    parser.add_argument("--protocol", choices=("legacy", "multiseed_v1", "multiseed_v2"), default="legacy")
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
