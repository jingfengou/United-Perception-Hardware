"""Localize early training divergence; replay is diagnostic, not acceptance."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import random
import runpy
import sys


def describe(value):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous()
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Unrecorded diagnostic value type: " + str(type(value)))


def rng_state():
    import numpy as np
    import torch

    return {"python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
            "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
            "torch_cpu": describe(torch.get_rng_state()),
            "torch_cuda": describe(torch.cuda.get_rng_state())}


class RoiTrace:
    """Observe the existing native call; an optional replay changes only its output."""

    def __init__(self, output, reference=None):
        self.output = output
        self.reference = reference
        self.step = None
        self.records = []

    def start(self, step):
        self.step = step
        self.records = []

    def wrap(self, native):
        import torch

        def backward(*args):
            if self.step is None:
                return native(*args)
            index = len(self.records)
            # The final argument is the zero-initialized destination gradient.
            inputs = describe(args[:-1])
            native(*args)
            observed = describe(args[-1])
            record = {"inputs": inputs, "native_output": observed}
            if self.reference is not None:
                path = self.reference / ("roi-{}-{}.pt".format(self.step, index))
                reference = torch.load(path, map_location="cpu", weights_only=True)
                if reference["inputs"] != inputs:
                    raise AssertionError("Replay inputs differ at step {}, ROI {}".format(self.step, index))
                args[-1].copy_(reference["gradient"].to(args[-1].device))
            record["used_output"] = describe(args[-1])
            self.records.append(record)
            torch.save({"inputs": inputs, "gradient": args[-1].detach().cpu()},
                       self.output / ("roi-{}-{}.pt".format(self.step, index)))
        return backward


def run(args):
    import torch

    helpers = runpy.run_path(str(Path(__file__).with_name("validate_training_accuracy.py")))
    helpers["prepare_legacy_environment"]()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty diagnostic output directory")
    output.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    data = json.loads(args.data.read_text())
    os.environ["DEFAULT_TASKS"] = "cls:det"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, str(source))
    import up
    initialize = vars(up).get("initialize")
    if initialize is not None and getattr(initialize, "__module__", None) == "up":
        initialize()
    if Path(up.__file__).resolve().parent != source / "up":
        raise RuntimeError("Imported the wrong UP tree")
    from up.runner.base_runner import BaseRunner
    from up.utils.general.global_flag import DIST_BACKEND

    DIST_BACKEND.backend = "dist"
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0)
    config, _ = helpers["effective_config"](source, data, "faster_rcnn", args.seed, 3)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["saver"].update(save_dir=str(output / "checkpoints"), results_dir=str(output / "predictions"))
    runner = BaseRunner(config, work_dir=str(output), training=True)
    report = {"scope": "diagnostic_only; early steps, not accuracy or full-run equivalence acceptance",
              "source": str(source), "driver_sha256": helpers["file_hash"](__file__),
              "seed": args.seed, "effective_config_sha256": config_hash,
              "data_sha256": helpers["file_hash"](args.data),
              "initial_state": helpers["state_hash"](runner.model.state_dict()),
              "replay_reference": str(args.replay_roi) if args.replay_roi else None,
              "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name(0),
                              "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                              "deterministic_algorithms": True, "tf32": False},
              "steps": []}
    report["initial_metrics"] = dict(runner.evaluate())
    roi = RoiTrace(output, args.replay_roi)
    native_roi = importlib.import_module("up.extensions._C").roi_align
    original_backward = native_roi.backward_avg_cuda
    native_roi.backward_avg_cuda = roi.wrap(original_backward)
    stage = {}

    def forward_hook(name):
        def capture(module, inputs, result):
            stage[name] = describe(result)
        return capture

    handles = [module.register_forward_hook(forward_hook(name))
               for name, module in runner.model.named_children()]
    try:
        runner._hooks("before_train")
        runner.model.cuda().train()
        for step in range(args.steps):
            roi.start(step)
            stage.clear()
            row = {"step": step, "rng_before_data": rng_state()}
            batch = runner.get_batch("train")
            row["batch"] = describe(batch)
            row["rng_before_forward"] = rng_state()
            loss = runner.forward_train(batch)
            row["forward_stages"] = dict(stage)
            row["loss"] = float(loss.detach())
            row["rng_after_forward"] = rng_state()
            runner.backward(loss)
            row["rng_after_backward"] = rng_state()
            row["roi_backward"] = roi.records
            gradients = {name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
                         for name, parameter in runner.model.named_parameters()}
            row["gradients"] = describe(gradients)
            torch.save(gradients, output / ("gradients-{}.pt".format(step)))
            runner.update()
            row["state_after_update"] = helpers["state_hash"](runner.model.state_dict())
            runner.lr_scheduler.step()
            row["rng_after_update"] = rng_state()
            report["steps"].append(row)
            print(json.dumps({"step": step, "loss": row["loss"], "roi_calls": len(roi.records)}), flush=True)
    finally:
        native_roi.backward_avg_cuda = original_backward
        for handle in handles:
            handle.remove()
    report["status"] = "completed"
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


def compare(args):
    import torch

    paths = args.compare
    reports = [json.loads((path / "report.json").read_text()) for path in paths]
    result = {"scope": "diagnostic_only; does not change the failed accuracy acceptance result",
              "left": str(paths[0]), "right": str(paths[1]), "steps": []}
    result["preconditions_equal"] = all(reports[0][key] == reports[1][key] for key in (
        "status", "seed", "effective_config_sha256", "data_sha256", "initial_state", "initial_metrics",
        "driver_sha256", "environment"))
    if len(reports[0]["steps"]) != len(reports[1]["steps"]):
        raise AssertionError("Different diagnostic step counts")
    for step, (left, right) in enumerate(zip(reports[0]["steps"], reports[1]["steps"])):
        row = {"step": step, "losses": [left["loss"], right["loss"]]}
        for key in ("batch", "rng_before_data", "rng_before_forward", "rng_after_forward",
                    "rng_after_backward", "rng_after_update", "state_after_update"):
            row[key + "_equal"] = left[key] == right[key]
        row["forward_stages_equal"] = {name: left["forward_stages"][name] == right["forward_stages"][name]
                                       for name in left["forward_stages"]}
        a, b = left["roi_backward"], right["roi_backward"]
        row["roi_call_count_equal"] = len(a) == len(b)
        row["roi_inputs_equal"] = [x["inputs"] == y["inputs"] for x, y in zip(a, b)]
        row["roi_native_outputs_equal"] = [x["native_output"] == y["native_output"] for x, y in zip(a, b)]
        row["roi_used_outputs_equal"] = [x["used_output"] == y["used_output"] for x, y in zip(a, b)]
        gradients = [torch.load(path / ("gradients-{}.pt".format(step)), map_location="cpu", weights_only=True)
                     for path in paths]
        differences = {}
        if gradients[0].keys() != gradients[1].keys():
            raise AssertionError("Different parameter keys")
        for name, value in gradients[0].items():
            other = gradients[1][name]
            if value is None or other is None:
                if value is not other:
                    differences[name] = {"none_mismatch": True}
            elif not torch.equal(value, other):
                differences[name] = {"max_abs": float((value.double() - other.double()).abs().max()),
                                     "different_elements": int((value != other).sum())}
        row["parameter_gradient_differences"] = differences
        row["gradients_bitwise_equal"] = not differences
        result["steps"].append(row)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({"preconditions_equal": result["preconditions_equal"], "steps": [
        {key: value for key, value in row.items() if key != "parameter_gradient_differences"}
        for row in result["steps"]]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--replay-roi", type=Path,
                        help="Diagnostic intervention only: validate and use recorded native backward outputs")
    parser.add_argument("--compare", type=Path, nargs=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.compare:
        compare(args)
    else:
        if not args.source or not args.data:
            parser.error("--source and --data are required for a run")
        if args.steps <= 0:
            parser.error("--steps must be positive")
        run(args)


if __name__ == "__main__":
    main()
