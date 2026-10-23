"""Audit full-budget checkpoints in fresh training and inference processes.

No model code, training recipe, numerical policy or tolerance is changed. Saved
validation batches let independent inference replay every actual eval input.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
import traceback

from validate_current_accuracy import BUDGETS, SEED
from validate_platform_equivalence import (
    MODELS, PROFILES, close_trees, file_hash, initialize, numerical_profile,
    predictions, routing, save_inference_weights, source_manifest, to_cpu,
    tree_hash, write_json,
)


def compare_metrics(expected, actual):
    if expected.keys() != actual.keys():
        raise AssertionError("Evaluation metric keys changed")
    for name, value in expected.items():
        if not math.isfinite(float(value)) or not math.isfinite(float(actual[name])):
            raise AssertionError("Non-finite metric: " + name)
        if abs(float(value) - float(actual[name])) > 1e-6:
            raise AssertionError("Evaluation metric changed: " + name)


def complete_training_report(path, model, backend, profile):
    report = json.loads((path / "report.json").read_text())
    epochs, protocol, steps = BUDGETS[model]
    expected = ("completed", model, backend, SEED, epochs, protocol, steps)
    actual = tuple(report[key] for key in (
        "status", "model", "backend", "seed", "epochs", "protocol", "training_steps"))
    if actual != expected or report["environment"]["numerical_profile"] != profile:
        raise ValueError("Not the complete predeclared training run/profile")
    return report


def state_snapshot(runner, accelerator, split):
    return {"model": tree_hash(runner.model.state_dict()),
            "optimizer": tree_hash(runner.optimizer.state_dict()),
            "scheduler": tree_hash(runner.lr_scheduler.state_dict()),
            "scaler": tree_hash(runner.scaler.state_dict()) if hasattr(runner, "scaler") else None,
            "gradients": tree_hash([p.grad for p in runner.model.parameters()]),
            "routing": routing(accelerator, split)}


def platform_source(args):
    actual = Path(sys.modules["up_platform"].__file__).resolve()
    expected = args.source / "packages/up-platform/src/up_platform"
    if actual.parent != expected.resolve():
        raise RuntimeError("Imported platform from a different source tree: " + str(actual))
    return str(actual)


def validate_reference(reference, args, environment):
    if (reference.get("passed"), reference.get("action"), reference.get("model"), reference.get("backend")) != (
            True, "training", args.model, args.backend):
        raise ValueError("Reference is not a passed restoration of the requested model/backend")
    if reference.get("environment") != environment:
        raise ValueError("Reference and independent inference environments differ")


def training(args):
    import torch
    accelerator, split, environment = initialize(args, "training")
    imported_platform = platform_source(args)
    from up.runner.base_runner import BaseRunner
    from up.utils.general.hook_helper import Hook
    original = complete_training_report(args.training, args.model, args.backend, numerical_profile(args))
    checkpoint_path = args.training / "comparison_state.pt"
    saved = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    config = json.loads((args.training / "effective_config.json").read_text())
    write_json(args.output / "effective_config.json", config)
    config = copy.deepcopy(config)
    config["saver"].update(auto_resume=False, save_dir=str(args.output / "checkpoints"),
                           results_dir=str(args.output / "results"))
    runner = BaseRunner(config, work_dir=str(args.output), training=True)
    runner.model.load_state_dict(saved["model"], strict=True)
    runner.optimizer.load_state_dict(saved["optimizer"])
    runner.lr_scheduler.load_state_dict(saved["scheduler"])
    assert tree_hash(saved["model"]) == tree_hash(runner.model.state_dict()), "Model restore changed state"
    assert tree_hash(saved["optimizer"]) == tree_hash(runner.optimizer.state_dict()), "Optimizer restore changed state"
    assert tree_hash(saved["scheduler"]) == tree_hash(runner.lr_scheduler.state_dict()), "Scheduler restore changed state"
    if saved.get("scaler") is not None:
        runner.scaler.load_state_dict(saved["scaler"])
        assert tree_hash(saved["scaler"]) == tree_hash(runner.scaler.state_dict()), "Scaler restore changed state"
    modules, parameters = list(runner.model.modules()), list(runner.model.parameters())
    modes = [m.training for m in modules]
    requires_grad = [p.requires_grad for p in parameters]
    before = state_snapshot(runner, accelerator, split)
    batch_folder = args.output / "validation-batches"
    batch_folder.mkdir()
    records, expected_outputs = [], []

    class Recorder(Hook):
        def before_eval_forward(self, cur_iter, batch):
            self.sample = {key: to_cpu(batch[key]) for key in (
                "image", "image_info", "gt", "gt_bboxes", "gt_ignores") if key in batch}

        def after_eval_forward(self, cur_iter, output):
            assert accelerator.purpose == "training" and not torch.is_grad_enabled()
            assert not runner.model.training
            value = predictions(output)
            path = batch_folder / ("{:04d}.pt".format(len(records)))
            torch.save({"input": self.sample, "prediction": value}, path)
            records.append({"file": str(path.relative_to(args.output)), "sha256": file_hash(path),
                            "input_sha256": tree_hash(self.sample), "prediction_sha256": tree_hash(value)})
            expected_outputs.append(value)

    runner._hooks.hooks.append(Recorder(runner))
    metrics = dict(runner.evaluate())
    assert records, "No validation input was evaluated"
    assert before == state_snapshot(runner, accelerator, split), "Eval updated training state/provider"
    assert modes == [m.training for m in modules], "Eval did not restore module modes"
    assert requires_grad == [p.requires_grad for p in parameters]
    assert all(a is b for a, b in zip(parameters, runner.model.parameters()))
    compare_metrics(original["final_metrics"], metrics)
    save_inference_weights(saved, args.output / "checkpoint.pt")
    torch.save(expected_outputs, args.output / "validation-predictions.pt")
    return {"environment": environment, "platform_source": imported_platform,
            "routing": routing(accelerator, split),
            "numerics": accelerator.numerics_state(), "validation_batches": records,
            "metrics": metrics, "checkpoint_restore": {"model": True, "optimizer": True, "scheduler": True,
                "scaler": "not used in the fixed FP32 protocol" if saved.get("scaler") is None else True},
            "module_modes_restored": True, "parameter_identity_preserved": True,
            "training_objects_unchanged": True, "state": before,
            "original_report_sha256": file_hash(args.training / "report.json"),
            "original_checkpoint_sha256": file_hash(checkpoint_path)}


def inference(args):
    import torch
    accelerator, split, environment = initialize(args, "inference")
    imported_platform = platform_source(args)
    from up.apis.inference import BaseInference
    from up.flags import ALIGNED_FLAG
    from up.utils.env.gene_env import to_device
    reference = json.loads((args.reference / "report.json").read_text())
    validate_reference(reference, args, environment)
    config = json.loads((args.reference / "effective_config.json").read_text())
    ALIGNED_FLAG.aligned = bool(config.get("runtime", {}).get("aligned", False))
    ALIGNED_FLAG.offset = 0 if ALIGNED_FLAG.aligned else 1
    data = json.loads(args.data.read_text())
    entry = data["cifar10" if args.model in ("resnet18", "vit_b16") else "pennfudan"]["test"]
    filename = (Path(entry["meta"]).read_text().splitlines()[0].rsplit(" ", 1)[0]
                if args.model in ("resnet18", "vit_b16") else
                json.loads(Path(entry["meta"]).read_text())["images"][0]["file_name"])
    checkpoint = args.reference / "checkpoint.pt"
    (args.output / "images").mkdir()
    config["args"] = {"image_path": str(Path(entry["image_dir"]) / filename), "ckpt": str(checkpoint),
                      "vis_dir": str(args.output / "images")}
    config["runtime"].setdefault("model_helper", {}).setdefault("kwargs", {})["cfg"] = config["net"]
    config["saver"] = {"auto_resume": False, "save_dir": str(args.output / "checkpoints"),
                       "results_dir": str(args.output / "results")}
    instance = BaseInference(config, {"type": "plt", "kwargs": {}}, work_dir=str(args.output))
    assert not any(hasattr(instance, name) for name in ("optimizer", "scaler"))
    before = tree_hash(instance.detector.state_dict())
    assert before == reference["state"]["model"], "Independent inference loaded different weights"
    image_outputs = []
    handle = instance.detector.register_forward_hook(
        lambda module, inputs, output: image_outputs.append(predictions(output)))
    instance.predict()
    handle.remove()
    assert len(image_outputs) == 1
    torch.save(image_outputs, args.output / "image-predictions.pt")
    actual_outputs, comparisons = [], []
    for record in reference["validation_batches"]:
        path = args.reference / record["file"]
        assert file_hash(path) == record["sha256"], "Saved validation batch changed"
        saved = torch.load(path, weights_only=False, map_location="cpu")
        with torch.no_grad():
            value = predictions(instance.detector(to_device(saved["input"], accelerator.device)))
        comparisons.append(close_trees(saved["prediction"], value))
        actual_outputs.append(value)
    assert tree_hash(instance.detector.state_dict()) == before, "Inference changed model state"
    assert accelerator.purpose == "inference" and not instance.detector.training
    assert not any(name in sys.modules for name in (
        "up_platform.training.backend", "up_platform.training.runtime.backend.base"))
    torch.save(actual_outputs, args.output / "validation-predictions.pt")
    return {"environment": environment, "platform_source": imported_platform,
            "routing": routing(accelerator, split),
            "numerics": accelerator.numerics_state(), "checkpoint_state_sha256": before,
            "predict_called": True, "image_predictions": 1, "training_objects_created": False,
            "training_modules_loaded": False, "training_eval_vs_inference": comparisons,
            "reference_report_sha256": file_hash(args.reference / "report.json"),
            "nvidia_binary_loaded": "up_platform.backends.nvidia._C" in sys.modules}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("training", "inference"))
    for name in ("source", "data", "output"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--training", type=lambda value: Path(value).resolve())
    parser.add_argument("--reference", type=lambda value: Path(value).resolve())
    parser.add_argument("--backend", choices=("nvidia", "metax"), required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--numerical-profile", choices=PROFILES, default="native")
    args = parser.parse_args()
    numerical_profile(args)
    if (args.action == "training" and not args.training) or (args.action == "inference" and not args.reference):
        parser.error("training needs --training; inference needs --reference")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        result = training(args) if args.action == "training" else inference(args)
        result.update(passed=True, action=args.action, model=args.model, backend=args.backend,
                      scope="restoration and all held-out batches of a complete single-seed fixed-budget run",
                      source=source_manifest(args.source), driver_sha256=file_hash(__file__),
                      elapsed_seconds=time.monotonic() - started)
        write_json(args.output / "report.json", result)
    except Exception:
        write_json(args.output / "report.json", {"passed": False, "error": traceback.format_exc(),
                                                 "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
