"""Bounded, single-seed real-device regression of the platform package split.

Each source tree runs in fresh processes. This is not the paused accuracy cohort.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


MODELS = ("resnet18", "yolov5s", "detr", "vit_b16")
SEED = 131
STEPS = 3
RTOL, ATOL = 1e-4, 1e-5


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def save_inference_weights(checkpoint, path):
    import torch
    # Deployment does not need optimizer/scheduler objects (including legacy NumPy scalars).
    torch.save({"model": checkpoint["model"]}, path)


def tree_hash(value):
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            digest.update(item.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def to_cpu(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(child) for child in value)
    return value


def predictions(output):
    keys = ("dt_bboxes",) if "dt_bboxes" in output else ("logits", "scores", "preds")
    result = {key: to_cpu(output[key]) for key in keys if key in output}
    if not result:
        raise ValueError("No recognized prediction output")
    return result


def source_manifest(root):
    files = {}
    for directory in ("up", "packages/up-platform/src/up_platform", "configs"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".cpp", ".cu", ".h", ".hpp", ".cuh", ".yaml", ".json"):
                files[path.relative_to(root).as_posix()] = hashlib.sha256(
                    path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return {"normalized_files": files, "sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()).hexdigest()}


def prepare_subset(manifest, output):
    """Select the first 8 train / 4 test images, preserving the original holdout."""
    from prepare_cross_vendor_data import verify_data
    original = json.loads(manifest.read_text())
    verify_data(original)
    data = copy.deepcopy(original)
    output.mkdir(parents=True, exist_ok=False)
    evidence = {}
    for name in ("cifar10", "pennfudan"):
        for split, count in (("train", 8), ("test", 4)):
            entry = data[name][split]
            source = Path(entry["meta"])
            target = output / (name + "-" + split + source.suffix)
            if name == "cifar10":
                lines = source.read_text().splitlines()[:count]
                if len(lines) != count:
                    raise ValueError("Not enough classification examples")
                target.write_text("\n".join(lines) + "\n")
                images = [line.rsplit(" ", 1)[0] for line in lines]
            else:
                annotation = json.loads(source.read_text())
                annotation["images"] = annotation["images"][:count]
                ids = {image["id"] for image in annotation["images"]}
                annotation["annotations"] = [item for item in annotation["annotations"] if item["image_id"] in ids]
                write_json(target, annotation)
                images = [image["file_name"] for image in annotation["images"]]
            image_hashes = {name: file_hash(Path(entry["image_dir"]) / name) for name in images}
            if name == "cifar10":
                records = [[filename, image_hashes[filename], int(label)]
                           for filename, label in (line.rsplit(" ", 1) for line in lines)]
            else:
                records = [[filename, image_hashes[filename], file_hash(
                    Path(entry["mask_dir"]) / (Path(filename).stem + "_mask.png"))] for filename in images]
            entry.update(meta=str(target), count=count, meta_sha256=file_hash(target))
            entry["indices"] = entry["indices"][:count]
            entry["image_manifest_sha256"] = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
            evidence[name + "/" + split] = {"metadata_sha256": file_hash(target), "images": image_hashes}
    verify_data(data)
    write_json(output / "data.json", data)
    write_json(output / "selection.json", {"selection": "first 8 train / 4 test; no resampling",
                                           "original_manifest_sha256": file_hash(manifest), "subsets": evidence})
    return output / "data.json"


def initialize(args, purpose):
    root = args.source.resolve()
    sys.path.insert(0, str(root))
    package = root / "packages/up-platform/src"
    split = package.is_dir()
    if split:
        sys.path.insert(0, str(package))
    from validate_training_accuracy import prepare_legacy_environment
    shims = prepare_legacy_environment()
    import torch
    from up.accelerator import configure_accelerator, get_accelerator
    options = {"backend": args.backend, "reset": True}
    if split:
        options["purpose"] = purpose
    accelerator = configure_accelerator(**options)
    accelerator.set_device(0)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    import up
    if Path(up.__file__).resolve().parent != root / "up":
        raise RuntimeError("Wrong UP source tree imported")
    up.initialize()
    from up.flags import DIST_BACKEND
    DIST_BACKEND.backend = "dist"
    from importlib.metadata import PackageNotFoundError, version
    versions = {}
    for name in ("torchvision", "numpy", "scipy", "timm", "einops", "pycocotools", "triton", "setuptools"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    environment = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "dependencies": versions, "compatibility_shims": shims,
                   "precision": {"profile": accelerator.numerics_state()["profile"],
                                 "matmul_precision": torch.get_float32_matmul_precision(),
                                 "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                                 "convolution_tf32": torch.backends.cudnn.allow_tf32,
                                 "convolution_library": torch.backends.cudnn.enabled,
                                 "deterministic": torch.are_deterministic_algorithms_enabled()}}
    assert environment["precision"]["profile"] == "native"
    if split:
        assert get_accelerator().purpose == purpose
    return accelerator, split, environment


def config_for(args):
    from validate_training_accuracy import effective_config
    data = json.loads(args.data.read_text())
    protocol = "multiseed_v1" if args.model == "resnet18" else "multiseed_v2"
    config, original = effective_config(args.source, data, args.model, SEED, 3, protocol)
    config["runtime"].update(iter_base=True, accelerator={"backend": args.backend, "numerics": "native"})
    config["trainer"].update(max_iter=STEPS, test_start=0, test_freq=2, save_freq=1000)
    config["saver"].update(save_dir=str(args.output / "checkpoints"), results_dir=str(args.output / "predictions"))
    for split in ("train", "test"):
        section = config["dataset"][split]
        section.setdefault("batch_sampler", copy.deepcopy(config["dataset"].get("batch_sampler", {})))
        section["batch_sampler"]["kwargs"]["batch_size"] = 2
        section.setdefault("dataloader", copy.deepcopy(config["dataset"].get("dataloader", {})))
        section["dataloader"]["kwargs"].update(num_workers=0)
    return config, original


def routing(accelerator, split):
    if not split:
        return {"purpose": "legacy_unseparated", "backend": accelerator.name}
    info = accelerator.capabilities()
    return {"purpose": info["purpose"], "backend": info["backend"],
            "providers": {name: [entry["owner"], entry["scope"]] for name, entry in info["operations"].items()}}


def train(args):
    import torch
    from validate_training_accuracy import batch_hash, rng_hash
    accelerator, split, environment = initialize(args, "training")
    from up.runner.base_runner import BaseRunner
    from up.utils.general.hook_helper import Hook
    config, original = config_for(args)
    write_json(args.output / "effective_config.json", config)
    runner = BaseRunner(config, work_dir=str(args.output), training=True)
    initial_hash = tree_hash(runner.model.state_dict())
    report = {"initial_state_sha256": initial_hash, "steps": [], "evaluations": []}

    class Recorder(Hook):
        def before_forward(self, cur_iter, batch):
            report["steps"].append({"batch_sha256": batch_hash(batch), "rng_sha256": rng_hash(),
                                    "model_training": runner.model.training})
            if split:
                assert accelerator.purpose == "training"

        def after_forward(self, cur_iter, output):
            report["steps"][-1]["losses"] = {name: float(value.detach().sum().cpu())
                                             for name, value in output.items() if "loss" in name}

        def after_backward(self, cur_iter, loss):
            gradients = {name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
                         for name, parameter in runner.model.named_parameters()}
            assert any(value is not None for value in gradients.values())
            assert all(torch.isfinite(value).all() for value in gradients.values() if value is not None)
            torch.save(gradients, args.output / ("gradients-{}.pt".format(len(report["steps"]))))
            report["steps"][-1].update(loss=float(loss.detach().cpu()), gradients_sha256=tree_hash(gradients))

        def after_update(self, cur_iter):
            report["steps"][-1]["state_sha256"] = tree_hash(runner.model.state_dict())

        def before_eval_forward(self, cur_iter, batch):
            if not (args.output / "inference-input.pt").exists():
                sample = {key: to_cpu(batch[key]) for key in ("image", "image_info", "gt", "gt_bboxes", "gt_ignores")
                          if key in batch}
                torch.save(sample, args.output / "inference-input.pt")

        def after_eval_forward(self, cur_iter, output):
            assert not torch.is_grad_enabled()
            assert not runner.model.training
            if split:
                assert accelerator.purpose == "training"
            eval_outputs.append(predictions(output))

    recorder = Recorder(runner)
    runner._hooks.hooks.append(recorder)
    original_evaluate, eval_outputs = runner.evaluate, []

    def audited_evaluate():
        modules = list(runner.model.modules())
        parameters = list(runner.model.parameters())
        modes = [module.training for module in modules]
        requires_grad = [parameter.requires_grad for parameter in parameters]
        def snapshot():
            return {"model": tree_hash(runner.model.state_dict()),
                    "optimizer": tree_hash(runner.optimizer.state_dict()),
                    "scaler": tree_hash(runner.scaler.state_dict()) if hasattr(runner, "scaler") else None,
                    "gradients": tree_hash([parameter.grad for parameter in parameters]),
                    "routing": routing(accelerator, split)}
        before = snapshot()
        eval_outputs.clear()
        metrics = original_evaluate()
        restored = modes == [module.training for module in modules]
        assert before == snapshot(), "Evaluation changed weights, gradients, optimizer, scaler or provider"
        assert requires_grad == [parameter.requires_grad for parameter in parameters]
        assert all(a is b for a, b in zip(parameters, runner.model.parameters())), "Parameter identity changed"
        if split:
            assert restored, "Training evaluation failed to restore module modes"
        else:
            # The old Runner leaves the model in eval; restore only caller state so
            # the same unmodified train loop can continue. Never change tensor data.
            for module, mode in zip(modules, modes):
                module.training = mode
        number = len(report["evaluations"]) + 1
        torch.save(list(eval_outputs), args.output / ("eval-{}.pt".format(number)))
        report["evaluations"].append({"after_steps": len(report["steps"]), "metrics": dict(metrics),
                                      "module_modes_restored_by_runner": restored,
                                      "legacy_caller_mode_restore": not split and not restored,
                                      "training_objects_unchanged": True, "routing": routing(accelerator, split)})
        return metrics

    runner.evaluate = audited_evaluate
    runner.train()
    assert len(report["steps"]) == STEPS and len(report["evaluations"]) == 2
    assert all(entry["model_training"] for entry in report["steps"])
    state = {"model": to_cpu(runner.model.state_dict()), "optimizer": to_cpu(runner.optimizer.state_dict()),
             "scheduler": to_cpu(runner.lr_scheduler.state_dict()),
             "scaler": to_cpu(runner.scaler.state_dict()) if hasattr(runner, "scaler") else None}
    torch.save(state, args.output / "checkpoint.pt")
    from up.models.model_helper import ModelHelper
    restored_model = ModelHelper(copy.deepcopy(config["net"])).to(accelerator.device).eval()
    restored_model.load_state_dict(state["model"], strict=True)
    assert tree_hash(restored_model.state_dict()) == tree_hash(state["model"])
    runner.model = restored_model
    restored_optimizer = runner.build_optimizer(copy.deepcopy(config["trainer"]["optimizer"]))
    restored_optimizer.load_state_dict(state["optimizer"])
    assert tree_hash(restored_optimizer.state_dict()) == tree_hash(state["optimizer"])
    sample = torch.load(args.output / "inference-input.pt", weights_only=False, map_location="cpu")
    from up.utils.env.gene_env import to_device
    with torch.no_grad():
        restored_output = predictions(restored_model(to_device(copy.deepcopy(sample), accelerator.device)))
    torch.save(restored_output, args.output / "restored-output.pt")
    close_trees(eval_outputs[0], restored_output)
    report.update(final_state_sha256=tree_hash(state["model"]), checkpoint_restore=True,
                  environment=environment, original_config=original, routing=routing(accelerator, split))
    return report


def infer(args):
    import torch
    accelerator, split, environment = initialize(args, "inference")
    from up.apis.inference import BaseInference
    training = args.training.resolve()
    checkpoint = torch.load(training / "checkpoint.pt", weights_only=False, map_location="cpu")
    weights = args.output / "weights-only.pt"
    save_inference_weights(checkpoint, weights)
    config = json.loads((training / "effective_config.json").read_text())
    from up.flags import ALIGNED_FLAG
    ALIGNED_FLAG.aligned = bool(config.get("runtime", {}).get("aligned", False))
    ALIGNED_FLAG.offset = 0 if ALIGNED_FLAG.aligned else 1
    data = json.loads(args.data.read_text())
    entry = data["cifar10" if args.model in ("resnet18", "vit_b16") else "pennfudan"]["test"]
    if args.model in ("resnet18", "vit_b16"):
        filename = Path(entry["meta"]).read_text().splitlines()[0].rsplit(" ", 1)[0]
    else:
        filename = json.loads(Path(entry["meta"]).read_text())["images"][0]["file_name"]
    config["args"] = {"image_path": str(Path(entry["image_dir"]) / filename),
                      "ckpt": str(weights), "vis_dir": str(args.output / "images")}
    config["runtime"].setdefault("model_helper", {}).setdefault("kwargs", {})["cfg"] = config["net"]
    config["saver"] = {"auto_resume": False, "save_dir": str(args.output / "checkpoints"),
                       "results_dir": str(args.output / "results")}
    (args.output / "images").mkdir()
    instance = BaseInference(config, {"type": "plt", "kwargs": {}}, work_dir=str(args.output))
    assert tree_hash(instance.detector.state_dict()) == tree_hash(checkpoint["model"])
    assert not any(hasattr(instance, name) for name in ("optimizer", "scaler"))
    images = []
    handle = instance.detector.register_forward_hook(lambda module, inputs, output: images.append(predictions(output)))
    instance.predict()
    handle.remove()
    assert len(images) == 1
    torch.save(images, args.output / "image-predictions.pt")
    sample = torch.load(training / "inference-input.pt", weights_only=False, map_location="cpu")
    from up.utils.env.gene_env import to_device
    with torch.no_grad():
        actual = predictions(instance.detector(to_device(copy.deepcopy(sample), accelerator.device)))
    expected = torch.load(training / "restored-output.pt", weights_only=False, map_location="cpu")
    consistency = close_trees(expected, actual)
    torch.save(actual, args.output / "batch-predictions.pt")
    if split:
        assert accelerator.purpose == "inference"
        assert "up_platform.backends.{}.training".format(args.backend) not in sys.modules
        assert "up_platform.training.backend" not in sys.modules
    return {"environment": environment, "routing": routing(accelerator, split), "predict_called": True,
            "image_predictions": len(images), "training_eval_vs_inference": consistency,
            "training_objects_created": False, "checkpoint_state_sha256": tree_hash(checkpoint["model"]),
            "weights_only_sha256": file_hash(weights),
            "original_training_checkpoint_sha256": file_hash(training / "checkpoint.pt"),
            "coordinate_convention": {"aligned": ALIGNED_FLAG.aligned, "offset": ALIGNED_FLAG.offset},
            "nvidia_binary_loaded": "up_platform.backends.nvidia._C" in sys.modules}


def close_trees(first, second, path="root"):
    import torch
    stats = {"tensor_count": 0, "exact_tensor_count": 0, "max_abs": 0., "max_rel": 0.}
    def visit(a, b, name):
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype, name
            torch.testing.assert_close(a, b, rtol=RTOL if a.is_floating_point() else 0,
                                       atol=ATOL if a.is_floating_point() else 0,
                                       msg=lambda detail: name + ": " + detail)
            stats["tensor_count"] += 1
            stats["exact_tensor_count"] += int(torch.equal(a, b))
            if a.is_floating_point() and a.numel():
                assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
                delta = (a.double() - b.double()).abs()
                stats["max_abs"] = max(stats["max_abs"], float(delta.max()))
                stats["max_rel"] = max(stats["max_rel"], float((delta / a.double().abs().clamp_min(1e-12)).max()))
        elif isinstance(a, dict):
            assert isinstance(b, dict) and a.keys() == b.keys(), name
            for key in a:
                visit(a[key], b[key], name + "/" + str(key))
        elif isinstance(a, (tuple, list)):
            assert type(a) is type(b) and len(a) == len(b), name
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, name + "/" + str(index))
        elif isinstance(a, float):
            assert abs(a - b) <= ATOL + RTOL * abs(a), name
        else:
            assert a == b, name
    visit(first, second, path)
    return stats


def compare_model(root, model):
    import torch
    old, new = root / "baseline" / model, root / "changed" / model
    reports = [json.loads((path / "training/report.json").read_text()) for path in (old, new)]
    a, b = reports
    assert a["environment"] == b["environment"], "Before/after environment mismatch"
    assert a["initial_state_sha256"] == b["initial_state_sha256"], "Initial weights differ"
    close_trees([row["losses"] for row in a["steps"]], [row["losses"] for row in b["steps"]])
    for first, second in zip(a["steps"], b["steps"]):
        assert first["batch_sha256"] == second["batch_sha256"], "Training batch differs"
        assert first["rng_sha256"] == second["rng_sha256"], "Training RNG state differs"
    compared, failures = {}, {}
    paths = ["training/checkpoint.pt", "training/restored-output.pt"]
    paths += ["training/gradients-{}.pt".format(i) for i in range(1, STEPS + 1)]
    paths += ["training/eval-{}.pt".format(i) for i in range(1, 3)]
    paths += ["inference/batch-predictions.pt", "inference/image-predictions.pt"]
    for path in paths:
        x = torch.load(old / path, weights_only=False, map_location="cpu")
        y = torch.load(new / path, weights_only=False, map_location="cpu")
        try:
            compared[path] = dict(passed=True, **close_trees(x, y))
        except AssertionError as exc:
            compared[path] = {"passed": False, "error": str(exc)}
            failures[path] = str(exc)
        del x, y
    for first, second in zip(a["evaluations"], b["evaluations"]):
        try:
            close_trees(first["metrics"], second["metrics"])
        except AssertionError as exc:
            failures["metrics-after-step-{}".format(first["after_steps"])] = str(exc)
    return {"model": model, "passed": not failures, "comparisons": compared, "failures": failures,
            "initial_state_sha256": a["initial_state_sha256"],
            "final_states_exact": a["final_state_sha256"] == b["final_state_sha256"],
            "losses_exact": [row["losses"] for row in a["steps"]] == [row["losses"] for row in b["steps"]],
            "training_batches_and_rng_exact": True,
            "new_eval_restores_modes": all(row["module_modes_restored_by_runner"] for row in b["evaluations"]),
            "legacy_eval_mode_restore_by_driver": any(row["legacy_caller_mode_restore"] for row in a["evaluations"])}


def suite(args):
    args.output.mkdir(parents=True, exist_ok=False)
    data = prepare_subset(args.data, args.output / "data")
    report = {"scope": "single-seed, 3 updates on 8 real training / 4 validation images per dataset; not accuracy convergence",
              "backend": args.backend, "seed": SEED, "steps": STEPS, "rtol": RTOL, "atol": ATOL,
              "models": {}, "failures": {}, "driver_sha256": file_hash(__file__),
              "sources": {name: source_manifest(path) for name, path in (("baseline", args.baseline), ("changed", args.changed))}}
    write_json(args.output / "protocol.json", {key: report[key] for key in ("scope", "backend", "seed", "steps", "rtol", "atol", "driver_sha256")})
    for model in MODELS:
        for label, source in (("baseline", args.baseline), ("changed", args.changed)):
            for phase in ("training", "inference"):
                output = args.output / label / model / phase
                command = [sys.executable, str(Path(__file__).resolve()), phase, "--source", str(source),
                           "--data", str(data), "--output", str(output), "--backend", args.backend, "--model", model]
                if phase == "inference":
                    command += ["--training", str(output.parent / "training")]
                output.parent.mkdir(parents=True, exist_ok=True)
                with (output.parent / (phase + ".log")).open("w") as log:
                    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=source).returncode
                print(label, model, phase, "PASS" if code == 0 else "FAIL", flush=True)
                if code:
                    report["failures"][label + "/" + model + "/" + phase] = code
                    break
        try:
            report["models"][model] = compare_model(args.output, model)
            if not report["models"][model]["passed"]:
                report["failures"][model + "/comparison"] = report["models"][model]["failures"]
        except Exception:
            report["failures"][model + "/comparison"] = traceback.format_exc()
        report["passed"] = not report["failures"]
        write_json(args.output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("suite", "training", "inference"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--changed", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("nvidia", "metax"), required=True)
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.data = args.data.resolve()
    if args.action == "suite":
        if not args.baseline or not args.changed:
            parser.error("suite needs --baseline and --changed")
        args.baseline, args.changed = args.baseline.resolve(), args.changed.resolve()
        report = suite(args)
        raise SystemExit(not report["passed"])
    if not args.source or not args.model or (args.action == "inference" and not args.training):
        parser.error("phase needs --source, --model and inference also --training")
    args.source = args.source.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        report = train(args) if args.action == "training" else infer(args)
        report.update(passed=True, model=args.model, action=args.action,
                      driver_sha256=file_hash(__file__), elapsed_seconds=time.monotonic() - started)
        write_json(args.output / "report.json", report)
    except Exception:
        write_json(args.output / "report.json", {"passed": False, "error": traceback.format_exc(),
                                                 "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
