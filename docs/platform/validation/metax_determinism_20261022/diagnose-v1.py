"""Replay actual training convolutions with fixed operands; never change model math."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

from validate_platform_equivalence import (ATOL, RTOL, config_for, initialize,
                                          source_manifest, to_cpu, tree_hash, write_json)


def difference(reference, actual):
    import torch
    if reference is None or actual is None:
        return {"exact": reference is actual, "max_abs": 0., "outside_tolerance": 0}
    assert reference.shape == actual.shape and reference.dtype == actual.dtype
    delta = (actual.double() - reference.double()).abs()
    return {"exact": torch.equal(reference, actual), "max_abs": float(delta.max()) if delta.numel() else 0.,
            "outside_tolerance": int((~torch.isclose(actual, reference, rtol=RTOL, atol=ATOL)).sum())}


def repeat_operation(operation, args, kwargs, repeats):
    """Run the stateless primitive, without a graph, RNG, loss or optimizer."""
    import torch
    first = operation(*args, **kwargs)
    reference = to_cpu(first)
    is_tensor = isinstance(first, torch.Tensor)
    reference_items = [reference] if is_tensor else list(reference)
    rows = []
    for index in range(repeats):
        actual = reference if index == 0 else to_cpu(operation(*args, **kwargs))
        values = [actual] if is_tensor else list(actual)
        rows.append([difference(a, b) for a, b in zip(reference_items, values, strict=True)])
    return first, {"repeats": repeats, "output_sha256": tree_hash(reference), "rows": rows,
                   "exact": all(item["exact"] for row in rows for item in row),
                   "max_abs": [max(row[i]["max_abs"] for row in rows) for i in range(len(reference_items))]}


def capture(args):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    accelerator, split, environment = initialize(args, "training")
    from up.runner.base_runner import BaseRunner
    config, _ = config_for(args)
    write_json(args.output / "effective_config.json", config)
    runner = BaseRunner(config, work_dir=str(args.output), training=True)
    if args.disable_dnn:
        accelerator.configure_convolution(enabled=False)
    names = {parameter.data_ptr(): name for name, parameter in runner.model.named_parameters()}
    records, saved = [], []

    class Trace(TorchDispatchMode):
        def __torch_dispatch__(self, operation, types, operands=(), kwargs=None):
            kwargs = kwargs or {}
            forward = operation is torch.ops.aten.convolution.default
            backward = operation is torch.ops.aten.convolution_backward.default
            if not (forward or backward):
                return operation(*operands, **kwargs)
            position = 2 if backward else 1
            weight = operands[position]
            before = tree_hash((operands, kwargs))
            result, stats = repeat_operation(operation, operands, kwargs, args.repeats if backward else 2)
            assert tree_hash((operands, kwargs)) == before, "Convolution changed its input tensors"
            item = {"operation": str(operation), "parameter": names.get(weight.data_ptr()),
                    "input_sha256": before, "tensor_shapes": [list(t.shape) for t in operands if isinstance(t, torch.Tensor)],
                    "parameters": to_cpu(operands[4:] if backward else operands[3:]), **stats}
            if backward and not stats["exact"] and len(saved) < 4:
                path = args.output / ("case-{}.pt".format(len(saved)))
                torch.save({"operation": str(operation), "args": to_cpu(operands), "kwargs": to_cpu(kwargs),
                            "parameter": item["parameter"], "input_sha256": before}, path)
                item["saved_case"] = path.name
                item["saved_case_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                saved.append(path.name)
            records.append(item)
            print(json.dumps({"parameter": item["parameter"], "operation": str(operation),
                              "exact": stats["exact"], "max_abs": stats["max_abs"]}), flush=True)
            return result

    runner.model.train()
    batch = runner.get_batch("train")
    initial = tree_hash(runner.model.state_dict())
    before_rng = tree_hash(accelerator.get_rng_state())
    with Trace():
        loss = runner.forward_train(batch)
        runner.backward(loss)
    return {"model": args.model, "environment": environment, "source": source_manifest(args.source),
            "initial_state_sha256": initial, "batch_sha256": tree_hash(batch), "rng_before_sha256": before_rng,
            "loss": float(loss.detach().cpu()), "rtol": RTOL, "atol": ATOL, "split": split,
            "convolution_library_enabled": torch.backends.cudnn.enabled, "records": records,
            "scope": "first real training batch; repeated primitive operands; no optimizer update",
            "completed": True}


def replay(args):
    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    case = torch.load(args.case, map_location="cpu", weights_only=False)
    assert tree_hash((case["args"], case["kwargs"])) == case["input_sha256"]

    def gpu(value):
        if isinstance(value, torch.Tensor):
            return value.to("cuda")
        if isinstance(value, (list, tuple)):
            return type(value)(gpu(item) for item in value)
        if isinstance(value, dict):
            return {key: gpu(item) for key, item in value.items()}
        return value

    operands, kwargs = gpu(case["args"]), gpu(case["kwargs"])
    namespace, name, overload = case["operation"].split(".")
    operation = getattr(getattr(getattr(torch.ops, namespace), name), overload)
    report = {"case": str(args.case), "input_sha256": case["input_sha256"], "parameter": case["parameter"],
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "rtol": RTOL, "atol": ATOL,
              "scope": "pure PyTorch, actual saved primitive operands; no UP or model imported", "providers": {}}
    for enabled in (True, False):
        torch.backends.cudnn.enabled = enabled
        _, stats = repeat_operation(operation, operands, kwargs, args.repeats)
        report["providers"]["dnn_on" if enabled else "dnn_off"] = stats
        print(json.dumps({"dnn": enabled, "exact": stats["exact"], "max_abs": stats["max_abs"]}), flush=True)
    assert "up" not in sys.modules and "up_platform" not in sys.modules
    report["completed"] = True
    return report


def regression(args):
    import validate_platform_equivalence as validation
    original_initialize, original_config = validation.initialize, validation.config_for

    def initialize_control(options, purpose):
        accelerator, split, environment = original_initialize(options, purpose)
        if args.disable_dnn:
            accelerator.configure_convolution(enabled=False)
            environment["precision"]["convolution_library"] = False
        return accelerator, split, environment

    def config_control(options):
        config, original = original_config(options)
        if args.disable_dnn:
            config["args"]["nocudnn"] = True
        return config, original

    validation.initialize, validation.config_for = initialize_control, config_control
    try:
        report = validation.train(args) if args.action == "training" else validation.infer(args)
        report.update(passed=True, diagnostic_control="dnn_off" if args.disable_dnn else "native")
        return report
    finally:
        validation.initialize, validation.config_for = original_initialize, original_config


def compare(args):
    import torch
    from validate_platform_equivalence import close_trees
    first = json.loads((args.first / "report.json").read_text())
    second = json.loads((args.second / "report.json").read_text())
    assert first["initial_state_sha256"] == second["initial_state_sha256"]
    assert [row["batch_sha256"] for row in first["steps"]] == [row["batch_sha256"] for row in second["steps"]]
    assert [row["rng_sha256"] for row in first["steps"]] == [row["rng_sha256"] for row in second["steps"]]
    result = {"rtol": RTOL, "atol": ATOL, "initial_batch_rng_identical": True, "artifacts": {}}
    for filename in ("checkpoint.pt", "gradients-1.pt", "gradients-2.pt", "gradients-3.pt",
                     "eval-1.pt", "eval-2.pt", "restored-output.pt"):
        a, b = [torch.load(path / filename, weights_only=False, map_location="cpu") for path in (args.first, args.second)]
        try:
            stats = dict(passed=True, **close_trees(a, b))
        except AssertionError as exc:
            stats = {"passed": False, "error": str(exc)}
        result["artifacts"][filename] = stats
    try:
        close_trees([row["losses"] for row in first["steps"]], [row["losses"] for row in second["steps"]])
        result["losses_passed"] = True
    except AssertionError as exc:
        result["losses_passed"] = False
        result["loss_error"] = str(exc)
    result["passed"] = result["losses_passed"] and all(item["passed"] for item in result["artifacts"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "replay", "training", "inference", "compare"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model", choices=("resnet18", "yolov5s", "detr"))
    parser.add_argument("--case", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="metax", choices=("metax",))
    parser.add_argument("--disable-dnn", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 2 or args.output.exists():
        parser.error("Need at least two repeats and a new output directory")
    if args.action in ("capture", "training", "inference") and not all((args.source, args.data, args.model)):
        parser.error("model execution needs --source, --data and --model")
    if args.action == "compare" and not all((args.first, args.second)):
        parser.error("compare needs --first and --second")
    if args.action == "replay" and not args.case:
        parser.error("replay needs --case")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    try:
        if args.action == "capture":
            report = capture(args)
        elif args.action == "replay":
            report = replay(args)
        elif args.action == "compare":
            report = compare(args)
        else:
            report = regression(args)
    except Exception:
        write_json(args.output / "failure.json", {"error": traceback.format_exc(),
                                                  "elapsed_seconds": time.monotonic() - started})
        raise
    report.update(elapsed_seconds=time.monotonic() - started,
                  tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
