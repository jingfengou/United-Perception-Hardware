"""Evaluation/state audit helpers for the fixed-budget DETR GEMM regression."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

from diagnose_detr_origin import cpu, digest, write_json


def audit_evaluations(runner, report, output):
    import torch
    from up.accelerator import get_accelerator
    from up.utils.general.hook_helper import Hook
    original = runner.evaluate
    observations = []
    report["evaluations"] = []

    class Observer(Hook):
        def after_eval_forward(self, cur_iter, result):
            assert not torch.is_grad_enabled() and not runner.model.training
            assert get_accelerator().purpose == "training"
            observations.append(cpu(result["dt_bboxes"]))

    runner._hooks.hooks.append(Observer(runner))

    def evaluate():
        model = runner.model
        parameters = list(model.parameters())
        modes = [module.training for module in model.modules()]
        gradient_flags = [p.requires_grad for p in parameters]

        def state():
            return {"model": digest(model.state_dict()), "optimizer": digest(runner.optimizer.state_dict()),
                    "gradients": digest([p.grad for p in parameters]),
                    "scaler": digest(runner.scaler.state_dict()) if hasattr(runner, "scaler") else None}

        before = state()
        observations.clear()
        metrics = original()
        assert before == state(), "Training evaluation changed training state"
        assert modes == [module.training for module in model.modules()]
        assert gradient_flags == [p.requires_grad for p in parameters]
        assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
        assert get_accelerator().purpose == "training"
        number = len(report["evaluations"]) + 1
        torch.save(list(observations), output / ("evaluation-{:02d}.pt".format(number)))
        report["evaluations"].append({"metrics": dict(metrics), "after_steps": len(report["steps"]),
                                      "predictions_sha256": digest(observations), "training_state_unchanged": True,
                                      "module_modes_restored": True, "parameter_identity_preserved": True,
                                      "purpose": "training", "state": before})
        return metrics

    runner.evaluate = evaluate


def finish_capture(runner, report, config, output):
    import torch
    from up.accelerator import get_accelerator
    from up.models.model_helper import ModelHelper
    report["final_metrics"] = dict(runner.evaluate())
    state = {"model": cpu(runner.model.state_dict()), "optimizer": cpu(runner.optimizer.state_dict()),
             "scheduler": cpu(runner.lr_scheduler.state_dict())}
    torch.save(state, output / "comparison_state.pt")
    restored = ModelHelper(copy.deepcopy(config["net"])).to(get_accelerator().device)
    restored.load_state_dict(state["model"], strict=True)
    assert digest(restored.state_dict()) == digest(state["model"])
    runner.model = restored
    runner.optimizer = runner.build_optimizer(copy.deepcopy(config["trainer"]["optimizer"]))
    runner.optimizer.load_state_dict(state["optimizer"])
    assert digest(runner.optimizer.state_dict()) == digest(state["optimizer"])
    report["restored_metrics"] = dict(runner.evaluate())
    assert report["final_metrics"] == report["restored_metrics"]
    assert report["evaluations"][-1]["predictions_sha256"] == report["evaluations"][-2]["predictions_sha256"]
    report["checkpoint_restore"] = True
    report["numerics"] = get_accelerator().numerics_state()
    report["audit_completed"] = True


def infer(args):
    import torch
    from diagnose_detr_origin import new_output, setup
    from validate_training_accuracy import prepare_legacy_environment
    from validate_platform_equivalence import close_trees
    source = args.source.resolve()
    sys.path[:0] = [str(source / "packages/up-platform/src"), str(source)]
    prepare_legacy_environment()
    config = json.loads((args.training / "effective_config.json").read_text())
    output = new_output(args.output)
    import up
    from up.accelerator import configure_accelerator
    ctx = configure_accelerator(config, purpose="inference", reset=True)
    up.initialize()
    setup()
    from up.flags import ALIGNED_FLAG, DIST_BACKEND
    DIST_BACKEND.backend = "dist"
    ALIGNED_FLAG.aligned = bool(config.get("runtime", {}).get("aligned", False))
    ALIGNED_FLAG.offset = 0 if ALIGNED_FLAG.aligned else 1
    state = torch.load(args.training / "comparison_state.pt", map_location="cpu", weights_only=False)["model"]
    if args.official:
        core = torch.load(args.official / "final.pt", map_location="cpu", weights_only=False)["model"]
        expected = {key.removeprefix("detr.detector.") for key in state if key.startswith("detr.detector.")}
        if expected != core.keys():
            raise ValueError("Official core checkpoint keys differ")
        state.update({"detr.detector." + key: value for key, value in core.items()})
    weights = output / "weights-only.pt"
    torch.save({"model": state}, weights)
    entry = config["dataset"]["test"]["dataset"]["kwargs"]
    first = json.loads(Path(entry["meta_file"]).read_text())["images"][0]["file_name"]
    config["args"] = {"image_path": str(Path(entry["image_reader"]["kwargs"]["image_dir"]) / first),
                      "ckpt": str(weights), "vis_dir": str(output / "images")}
    config["runtime"].setdefault("model_helper", {}).setdefault("kwargs", {})["cfg"] = config["net"]
    config["saver"] = {"auto_resume": False, "save_dir": str(output / "checkpoints"),
                       "results_dir": str(output / "results")}
    (output / "images").mkdir()
    from up.apis.inference import BaseInference
    instance = BaseInference(config, {"type": "plt", "kwargs": {}}, work_dir=str(output))
    assert digest(instance.detector.state_dict()) == digest(state)
    assert not any(hasattr(instance, name) for name in ("optimizer", "scaler"))
    instance.predict()
    from up.data.data_builder import BaseDataLoaderBuilder
    from up.utils.env.gene_env import to_device
    loader = BaseDataLoaderBuilder(copy.deepcopy(config["dataset"]), "test:test").build_dataloader()
    iterator, predictions, dumps = iter(loader), [], []
    seen_images = 0
    with torch.no_grad():
        for _ in range(loader.get_epoch_size()):
            batch = to_device(next(iterator), ctx.device)
            result = instance.detector(batch)
            predictions.append(cpu(result["dt_bboxes"]))
            dumps.append(loader.dataset.dump(result))
            seen_images += len(batch["image_info"])
    assert seen_images == len(loader.dataset) == 50
    reference_report = json.loads((args.training / "report.json").read_text())
    number = len(reference_report["evaluations"]) - 1
    expected = torch.load(args.training / ("evaluation-{:02d}.pt".format(number)), map_location="cpu", weights_only=True)
    comparison = close_trees(expected, predictions)
    torch.save(predictions, output / "predictions.pt")
    result_file = output / "results.jsonl"
    with result_file.open("x") as stream:
        for rows in dumps:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    metrics = dict(loader.dataset.evaluate(str(result_file), [dumps]))
    for name, value in reference_report["final_metrics"].items():
        assert abs(metrics[name] - value) <= 1e-6, name
    assert ctx.purpose == "inference"
    assert "up_platform.training.backend" not in sys.modules
    assert "up_platform.backends.nvidia._C" not in sys.modules
    write_json(output / "report.json", {"passed": True, "purpose": ctx.purpose, "predict_called": True,
                                       "seen_images": seen_images, "metrics": metrics, "checkpoint_model_sha256": digest(state),
                                       "eval_vs_inference": comparison, "predictions_sha256": digest(predictions),
                                       "numerics": ctx.numerics_state(), "rtol": 1e-4, "atol": 1e-5,
                                       "metric_atol": 1e-6, "optimizer_or_scaler_created": False,
                                       "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--official", type=Path, help="Load independent official core weights for the same evaluation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("DEFAULT_TASKS", "cls:det")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    infer(args)


if __name__ == "__main__":
    main()
