"""Separate UP orchestration from pinned official DETR and fixed ATen operands.

Capture uses the frozen UP recipe. Standalone consumes the actual tensor tape,
not a newly sampled loader. Native standalone/replay forbid importing either UP
package. A local-convolution control reuses the frozen, opt-in platform mode.
"""
import argparse
import contextlib
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

RTOL, ATOL = 1e-4, 1e-5


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def digest(value):
    import torch
    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            result.update(str((str(item.dtype), tuple(item.shape))).encode())
            result.update(item.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                result.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            result.update(pickle.dumps(item))
    visit(value)
    return result.hexdigest()


def map_tensors(value, function):
    import torch
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, dict):
        return {key: map_tensors(item, function) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(map_tensors(item, function) for item in value)
    return value


def cpu(value):
    return map_tensors(value, lambda tensor: tensor.detach().cpu().clone())


def rng():
    import numpy as np
    import torch
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2].cpu())
    torch.cuda.set_rng_state(state[3].cpu())


def setup():
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


class BlockUP(importlib.abc.MetaPathFinder):
    def __init__(self, platform=False):
        self.names = ("up", "up_platform") if platform else ("up",)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.names:
            raise ImportError("Standalone diagnostic forbids importing " + fullname)


def gradients(model):
    return {name: parameter.grad for name, parameter in model.named_parameters() if parameter.requires_grad}


def tensor_comparison(reference, actual):
    import torch
    if reference is None or actual is None:
        return {"exact": reference is actual, "outside_tolerance": int(reference is not actual), "max_abs": 0.}
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        raise ValueError("Tensor metadata differs")
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        raise ValueError("Nonfinite diagnostic tensor")
    delta = (reference.double() - actual.double()).abs()
    return {"exact": torch.equal(reference, actual),
            "outside_tolerance": int((~torch.isclose(actual, reference, rtol=RTOL, atol=ATOL)).sum()),
            "max_abs": float(delta.max()) if delta.numel() else 0.}


def compare_gradients(reference, actual):
    if reference.keys() != actual.keys():
        raise ValueError("Different parameter gradient keys")
    differences = {name: tensor_comparison(value, actual[name]) for name, value in reference.items()}
    return {"exact": all(row["exact"] for row in differences.values()),
            "outside_tolerance": sum(row["outside_tolerance"] for row in differences.values()),
            "differences": {name: row for name, row in differences.items() if not row["exact"]}}


def new_output(path):
    path.mkdir(parents=True, exist_ok=False)
    return path


def capture(args):
    import copy
    import torch
    from validate_training_accuracy import effective_config, prepare_legacy_environment

    output = new_output(args.output)
    prepare_legacy_environment()
    source = args.source.resolve()
    sys.path[:0] = [str(source / "packages/up-platform/src"), str(source)]
    import up
    from up.accelerator import configure_accelerator
    configure_accelerator({"runtime": {"accelerator": {"backend": "metax", "numerics": args.profile}}})
    up.initialize()
    from up.runner.base_runner import BaseRunner
    from up.utils.general.global_flag import DIST_BACKEND
    from up.utils.general.hook_helper import Hook

    setup()
    DIST_BACKEND.backend = "dist"
    data = json.loads(args.data.read_text())
    config, _ = effective_config(source, data, "detr", 131, 12, "multiseed_v2")
    config["runtime"]["accelerator"] = {"backend": "metax", "numerics": args.profile}
    write_json(output / "effective_config.json", config)
    config["saver"].update(save_dir=str(output / "checkpoints"), results_dir=str(output / "predictions"))
    runner = BaseRunner(config, work_dir=str(output), training=True)
    adapter = dict(runner.model.named_modules())["detr"]
    model = adapter.detector
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = [{**{key: value for key, value in group.items() if key != "params"},
               "params": [names[id(parameter)] for parameter in group["params"]]}
              for group in runner.optimizer.param_groups]
    initial = {"model": cpu(model.state_dict()), "criterion": cpu(adapter.criterion.state_dict()),
               "optimizer": cpu(runner.optimizer.state_dict()), "parameter_groups": groups,
               "requires_grad": {name: p.requires_grad for name, p in model.named_parameters()},
               "detr": data["detr"], "profile": args.profile, "seed": 131,
               "config": copy.deepcopy(config)}
    torch.save(initial, output / "initial.pt")
    report = {"scope": "diagnostic tensor tape; original 720-update schedule; no model or tolerance changes",
              "source": str(source), "profile": args.profile, "steps": [], "rtol": RTOL, "atol": ATOL,
              "initial_model_sha256": digest(initial["model"]), "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    tape, current = {}, {}
    handles = []

    def detector_input(module, inputs):
        if not module.training:
            return
        tape["images"] = cpu(inputs[0].tensors)
        tape["mask"] = cpu(inputs[0].mask)

    def criterion_input(module, inputs):
        if model.training:
            tape["targets"] = cpu(inputs[1])

    handles.extend([model.register_forward_pre_hook(detector_input),
                    adapter.criterion.register_forward_pre_hook(criterion_input)])

    class Before(Hook):
        def before_forward(self, cur_iter, batch):
            tape.clear()
            current.clear()
            index = len(report["steps"])
            tape.update(rng=rng(), step=index + 1,
                        lr=[float(group["lr"]) for group in runner.optimizer.param_groups])
            if index in (0, 56, 57, 58, 59):
                torch.save({"model": cpu(model.state_dict()), "optimizer": cpu(runner.optimizer.state_dict())},
                           output / ("before-{:04d}.pt".format(index + 1)))

        def after_forward(self, cur_iter, result):
            current["loss_terms"] = {name: float(value.detach()) for name, value in result.items() if "loss" in name}

        def after_backward(self, cur_iter, loss):
            raw = cpu(gradients(model))
            current.update(step=tape["step"], loss=float(loss.detach()), raw_gradient_sha256=digest(raw),
                           parameter_gradients={name: digest(value) for name, value in raw.items()})
            if tape["step"] <= args.save_gradients:
                torch.save(raw, output / ("gradients-{:04d}.pt".format(tape["step"])))

    class After(Hook):
        def after_backward(self, cur_iter, loss):
            current["clipped_gradient_sha256"] = digest(gradients(model))

        def after_update(self, cur_iter):
            current.update(weight_sha256=digest(model.state_dict()), input_sha256=digest(
                {key: tape[key] for key in ("images", "mask", "targets")}), lr=tape["lr"])
            torch.save(tape, output / ("batch-{:04d}.pt".format(tape["step"])))
            report["steps"].append(dict(current))
            write_json(output / "report.json", report)
            print(json.dumps({key: current[key] for key in ("step", "loss", "raw_gradient_sha256")}), flush=True)

    runner._hooks.hooks.insert(0, Before(runner))
    runner._hooks.hooks.append(After(runner))
    report["initial_metrics"] = dict(runner.evaluate())
    if args.steps < 720:
        runner.max_iter = args.steps
    started = time.monotonic()
    try:
        runner.train()
    finally:
        for handle in handles:
            handle.remove()
    report.update(completed=True, training_steps=len(report["steps"]), elapsed_seconds=time.monotonic() - started)
    write_json(output / "report.json", report)
    torch.save({"model": cpu(model.state_dict()), "optimizer": cpu(runner.optimizer.state_dict())}, output / "final.pt")


def load_official(initial, criterion):
    import importlib
    import torch
    from accuracy_detr_adapter import criterion_class, file_hash, source_hash
    from torchvision.models import resnet50

    metadata = initial["detr"]
    source = Path(metadata["source"]).resolve()
    if source_hash(source) != metadata["source_sha256"] or file_hash(metadata["checkpoint"]) != metadata["checkpoint_sha256"]:
        raise ValueError("Pinned official DETR source/checkpoint changed")
    sys.path.insert(0, str(source))
    detr = importlib.import_module("models.detr")
    backbone = importlib.import_module("models.backbone")
    transformer = importlib.import_module("models.transformer")
    position = importlib.import_module("models.position_encoding")
    matcher = importlib.import_module("models.matcher")
    misc = importlib.import_module("util.misc")
    if Path(detr.__file__).resolve().parent != source / "models":
        raise ValueError("Wrong official DETR imported")
    # The official Backbone constructor downloads weights. Build its exact classes
    # offline instead; all parameters are then restored from the captured state.
    body = resnet50(weights=None, replace_stride_with_dilation=[False, False, False], norm_layer=backbone.FrozenBatchNorm2d)
    features = backbone.BackboneBase(body, train_backbone=True, num_channels=2048, return_interm_layers=False)
    features = backbone.Joiner(features, position.PositionEmbeddingSine(128, normalize=True))
    features.num_channels = 2048
    model = detr.DETR(features, transformer.Transformer(
        d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
        dim_feedforward=2048, dropout=.1, return_intermediate_dec=True),
        num_classes=1, num_queries=100, aux_loss=True)
    weights = {"loss_ce": 1., "loss_bbox": 5., "loss_giou": 2.}
    weights.update({name + "_" + str(index): value for index in range(5) for name, value in list(weights.items())})
    base = detr.SetCriterion if criterion == "official" else criterion_class(detr.SetCriterion, misc.accuracy)
    loss = base(1, matcher.HungarianMatcher(1, 5, 2), weights, .1, ["labels", "boxes", "cardinality"])
    model.load_state_dict(initial["model"], strict=True)
    loss.load_state_dict(initial["criterion"], strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(initial["requires_grad"][name])
    return model, loss, misc.NestedTensor


def local_mode(args):
    if args.profile == "native":
        return contextlib.nullcontext()
    if args.source is None:
        raise ValueError("Local-convolution control needs its frozen platform source")
    sys.path.insert(0, str(args.source.resolve() / "packages/up-platform/src"))
    from up_platform.backends.metax.determinism import DeterministicConvolution
    return DeterministicConvolution()


def standalone(args):
    import torch
    sys.meta_path.insert(0, BlockUP(platform=args.profile == "native"))
    setup()
    output = new_output(args.output)
    initial = torch.load(args.tape / "initial.pt", map_location="cpu", weights_only=False)
    reference = json.loads((args.tape / "report.json").read_text())
    if not reference.get("completed") or len(reference["steps"]) < args.steps:
        raise ValueError("Requested steps exceed the completed reference tape")
    model, criterion, nested = load_official(initial, args.criterion)
    model.to("cuda").train()
    criterion.to("cuda").train()
    parameters = dict(model.named_parameters())
    groups = [{**group, "params": [parameters[name] for name in group["params"]]} for group in initial["parameter_groups"]]
    optimizer = torch.optim.AdamW(groups)
    optimizer.load_state_dict(initial["optimizer"])
    report = {"scope": "official DETR core, frozen tensor/RNG/LR tape; no UP orchestration",
              "profile": args.profile, "criterion": args.criterion, "steps": [], "rtol": RTOL, "atol": ATOL,
              "initial_model_sha256": digest(model.state_dict()), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(0), "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if report["initial_model_sha256"] != reference["initial_model_sha256"]:
        raise ValueError("Initial official model differs")
    with local_mode(args):
        for index, ref in enumerate(reference["steps"][:args.steps]):
            tape = torch.load(args.tape / ("batch-{:04d}.pt".format(index + 1)), map_location="cpu", weights_only=False)
            data = map_tensors({key: tape[key] for key in ("images", "mask", "targets")}, lambda t: t.to("cuda"))
            restore_rng(tape["rng"])
            for group, lr in zip(optimizer.param_groups, tape["lr"], strict=True):
                group["lr"] = lr
            prediction = model(nested(data["images"], data["mask"]))
            terms = criterion(prediction, data["targets"])
            losses = {name: value * criterion.weight_dict[name] for name, value in terms.items() if name in criterion.weight_dict}
            total = sum(losses.values())
            model.zero_grad()
            total.backward()
            raw = cpu(gradients(model))
            row = {"step": index + 1, "loss": float(total.detach()), "raw_gradient_sha256": digest(raw),
                   "parameter_gradients": {name: digest(value) for name, value in raw.items()},
                   "loss_terms": {name: float(value.detach()) for name, value in losses.items()},
                   "input_sha256": digest(data), "rng_after_backward_sha256": digest(rng())}
            if row["input_sha256"] != ref["input_sha256"]:
                raise ValueError("Standalone inputs differ from the captured official tensors")
            row["reference_raw_gradient_exact"] = row["raw_gradient_sha256"] == ref["raw_gradient_sha256"]
            saved = args.tape / ("gradients-{:04d}.pt".format(index + 1))
            if saved.exists() and not row["reference_raw_gradient_exact"]:
                row["reference_raw_gradient_comparison"] = compare_gradients(torch.load(saved, weights_only=True), raw)
                torch.save(raw, output / ("gradients-{:04d}.pt".format(index + 1)))
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad and p.grad is not None], .1, 2)
            row["clipped_gradient_sha256"] = digest(gradients(model))
            optimizer.step()
            row["weight_sha256"] = digest(model.state_dict())
            row["reference_weight_exact"] = row["weight_sha256"] == ref["weight_sha256"]
            report["steps"].append(row)
            write_json(output / "report.json", report)
            print(json.dumps({key: row[key] for key in ("step", "loss", "reference_raw_gradient_exact", "reference_weight_exact")}), flush=True)
    report.update(completed=True, up_imported=any(n == "up" or n.startswith("up.") for n in sys.modules),
                  platform_imported=any(n == "up_platform" or n.startswith("up_platform.") for n in sys.modules))
    assert not report["up_imported"]
    if args.profile == "native":
        assert not report["platform_imported"]
    torch.save({"model": cpu(model.state_dict()), "optimizer": cpu(optimizer.state_dict())}, output / "final.pt")
    write_json(output / "report.json", report)


def tensor_layout(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype),
                "device": str(value.device)}
    if isinstance(value, dict):
        return {key: tensor_layout(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [tensor_layout(item) for item in value]
    return repr(value)


def trace_step(args):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    sys.meta_path.insert(0, BlockUP(platform=args.profile == "native"))
    setup()
    output = new_output(args.output)
    initial = torch.load(args.tape / "initial.pt", map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.tape / ("before-{:04d}.pt".format(args.step)), map_location="cpu", weights_only=False)
    tape = torch.load(args.tape / ("batch-{:04d}.pt".format(args.step)), map_location="cpu", weights_only=False)
    model, criterion, nested = load_official(initial, args.criterion)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to("cuda").train()
    criterion.to("cuda").train()
    data = map_tensors({key: tape[key] for key in ("images", "mask", "targets")}, lambda t: t.to("cuda"))
    reference = json.loads(args.reference.read_text())["operations"] if args.reference else None
    report = {"scope": "single frozen official DETR step; trace returns native results unchanged",
              "step": args.step, "profile": args.profile, "criterion": args.criterion,
              "model_sha256": digest(model.state_dict()), "input_sha256": digest(data),
              "rng_sha256": digest(tape["rng"]), "operations": [], "first_difference": None}

    class Trace(TorchDispatchMode):
        def __torch_dispatch__(self, operation, types, operands=(), kwargs=None):
            kwargs = kwargs or {}
            name = str(operation)
            uninitialized = name.startswith(("aten.empty.", "aten.empty_like.", "aten.empty_strided.", "aten.new_empty."))
            writes_destination = name.startswith(("aten.zero_.", "aten.fill_.", "aten.copy_.", "aten.bernoulli_.", "aten.uniform_.", "aten.normal_."))
            observed_inputs = operands[1:] if writes_destination else operands
            row = {"index": len(report["operations"]), "operation": str(operation),
                   "input_sha256": digest((observed_inputs, kwargs)), "input_layout": tensor_layout((operands, kwargs))}
            result = operation(*operands, **kwargs)
            row.update(output_sha256="uninitialized allocation" if uninitialized else digest(result),
                       output_layout=tensor_layout(result))
            if reference is not None and report["first_difference"] is None:
                index = row["index"]
                ref = reference[index] if index < len(reference) else None
                if ref is None or any(row[key] != ref[key] for key in ("operation", "input_sha256", "output_sha256")):
                    report["first_difference"] = {"actual": row, "reference": ref}
                    torch.save({"operation": str(operation), "args": cpu(operands), "kwargs": cpu(kwargs),
                                "layout": tensor_layout((operands, kwargs)), "output": cpu(result),
                                "input_sha256": row["input_sha256"], "mutable": operation._schema.is_mutable},
                               output / "first-different-operation.pt")
            report["operations"].append(row)
            return result

    restore_rng(tape["rng"])
    # Trace below the optional convolution mode also observes its native suboperations.
    try:
        with Trace(), local_mode(args):
            predictions = model(nested(data["images"], data["mask"]))
            terms = criterion(predictions, data["targets"])
            total = sum(value * criterion.weight_dict[name] for name, value in terms.items() if name in criterion.weight_dict)
            total.backward()
    except Exception as error:
        report.update(completed=False, error=repr(error),
                      up_imported=any(n == "up" or n.startswith("up.") for n in sys.modules),
                      platform_imported=any(n == "up_platform" or n.startswith("up_platform.") for n in sys.modules))
        write_json(output / "trace.json", report)
        raise
    report.update(loss=float(total.detach()), gradient_sha256=digest(gradients(model)), completed=True,
                  up_imported=any(n == "up" or n.startswith("up.") for n in sys.modules),
                  platform_imported=any(n == "up_platform" or n.startswith("up_platform.") for n in sys.modules))
    assert not report["up_imported"]
    case_path = output / "first-different-operation.pt"
    if case_path.exists():
        observed = torch.load(case_path, map_location="cpu", weights_only=False)["output"]
        if isinstance(observed, torch.Tensor):
            report["matching_parameter_gradients"] = [
                name for name, value in gradients(model).items() if value is not None
                and value.shape == observed.shape and torch.equal(value.detach().cpu(), observed)]
    if reference is not None:
        report["operation_count_equal"] = len(reference) == len(report["operations"])
    write_json(output / "trace.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "operations"}), flush=True)


def restore_layout(value, layout):
    import torch
    if isinstance(value, torch.Tensor):
        if not str(layout["device"]).startswith("cuda"):
            return value
        shape, stride = layout["shape"], layout["stride"]
        extent = 0 if any(size == 0 for size in shape) else 1 + sum((size - 1) * step for size, step in zip(shape, stride))
        storage = torch.empty(extent, dtype=value.dtype, device="cuda")
        result = storage.as_strided(shape, stride)
        source, target = value, result
        for axis, (size, step) in enumerate(zip(shape, stride)):
            if step == 0 and size > 1:
                source = source.narrow(axis, 0, 1)
                target = target.narrow(axis, 0, 1)
        target.copy_(source)
        return result
    if isinstance(value, dict):
        return {key: restore_layout(item, layout[key]) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(restore_layout(item, meta) for item, meta in zip(value, layout, strict=True))
    return value


def replay(args):
    import torch
    sys.meta_path.insert(0, BlockUP(platform=True))
    setup()
    output = new_output(args.output)
    case = torch.load(args.case, map_location="cpu", weights_only=False)
    if case["mutable"]:
        raise ValueError("A mutable operation needs a dedicated state-preserving reproduction")
    operands, kwargs = restore_layout((case["args"], case["kwargs"]), case["layout"])
    if digest((operands, kwargs)) != case["input_sha256"]:
        raise ValueError("Replay operands changed")
    namespace, name, overload = case["operation"].split(".")
    operation = getattr(getattr(getattr(torch.ops, namespace), name), overload)
    reports = []
    reference = None
    for _ in range(args.repeats):
        result = cpu(operation(*operands, **kwargs))
        if reference is None:
            reference = result
            torch.save(reference, output / "first-result.pt")
        # ATen results may contain requested and unrequested gradients.
        left = reference if isinstance(reference, (tuple, list)) else [reference]
        right = result if isinstance(result, (tuple, list)) else [result]
        reports.append([tensor_comparison(a, b) for a, b in zip(left, right, strict=True)])
    assert digest((operands, kwargs)) == case["input_sha256"]
    assert not any(n.split(".")[0] in ("up", "up_platform") for n in sys.modules)
    report = {"scope": "pure PyTorch fixed ATen operands; no UP/platform/model/loss/optimizer",
              "operation": case["operation"], "input_sha256": case["input_sha256"], "layout": case["layout"],
              "repeats": args.repeats, "rtol": RTOL, "atol": ATOL, "rows": reports,
              "exact": all(row["exact"] for rows in reports for row in rows),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "completed": True}
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "standalone", "trace_step", "replay"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--tape", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("native", "metax_deterministic_v1"), default="metax_deterministic_v1")
    parser.add_argument("--criterion", choices=("matched", "official"), default="matched")
    parser.add_argument("--steps", type=int, default=70)
    parser.add_argument("--save-gradients", type=int, default=70)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.steps <= 720:
        parser.error("Steps must remain within the frozen 720-update schedule")
    if args.repeats < 2:
        parser.error("A repeatability control requires at least two repetitions")
    if args.action == "capture" and (args.source is None or args.data is None):
        parser.error("Capture needs --source and --data")
    if args.action in ("standalone", "trace_step") and args.tape is None:
        parser.error("Standalone/trace needs --tape")
    if args.action == "replay" and args.case is None:
        parser.error("Replay needs --case")
    os.environ.setdefault("DEFAULT_TASKS", "cls:det")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    globals()[args.action](args)


if __name__ == "__main__":
    main()
