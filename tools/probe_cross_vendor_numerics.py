"""Isolate RNG, early training, and operator-provider differences (not accuracy)."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import sys


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure():
    import torch
    os.environ["DEFAULT_TASKS"] = "cls:det"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def rng_state():
    import numpy as np
    import torch
    device_state = (sys.modules["up.accelerator"].get_accelerator().get_rng_state()
                    if "up.accelerator" in sys.modules else torch.cuda.get_rng_state())
    values = {"python": random.getstate(), "numpy": np.random.get_state(),
              "cpu": torch.get_rng_state().numpy(), "gpu": device_state.cpu().numpy()}
    return {key: hashlib.sha256(pickle.dumps(value)).hexdigest() for key, value in values.items()}


def rng_probe(args):
    import torch
    import torch.nn.functional as F
    tensors, rows = {}, {}
    for device in ("cpu", "cuda"):
        for seed in (131, 132):
            for shape in ((64, 1, 1), (64, 17, 768), (64, 12, 17, 17), (100, 2, 256)):
                for kind in ("rand", "dropout"):
                    torch.manual_seed(seed)
                    if args.numerical_profile != "native":
                        sys.modules["up.accelerator"].get_accelerator().manual_seed(seed)
                    name = "{}-{}-{}-{}".format(device, seed, shape, kind)
                    before = rng_state()
                    x = (torch.rand(shape, device=device) if kind == "rand" else
                         F.dropout(torch.ones(shape, device=device), p=.1, training=True))
                    tensors[name] = x.cpu()
                    rows[name] = {"before": before, "after": rng_state(), "zeros": int((x == 0).sum())}
    if args.rng_wide:
        for seed in (131, 132):
            torch.manual_seed(seed)
            if args.numerical_profile != "native":
                sys.modules["up.accelerator"].get_accelerator().manual_seed(seed)
            shape = (64, 17, 3072)
            before = rng_state()
            x = F.dropout(torch.ones(shape, device="cuda"), p=.1, training=True)
            name = "cuda-{}-wide-dropout-mask".format(seed)
            tensors[name] = (x == 0).cpu()
            rows[name] = {"before": before, "after": rng_state(), "zeros": int((x == 0).sum())}
    torch.save(tensors, args.output / "tensors.pt")
    return {"cases": rows}


def operator_probe(args):
    import torch
    sys.path.insert(0, str(args.source.resolve()))
    from up.accelerator import configure_accelerator, ops
    configure_accelerator(backend=args.backend)
    generator = torch.Generator(device="cpu").manual_seed(131)
    boxes = torch.rand((37, 4), generator=generator) * 100
    boxes[:, 2:4] += boxes[:, :2]
    boxes[:3] = torch.tensor([[0., 0., 0., 0.], [0., 0., -.5, -.5], [0., 0., 10., 10.]])
    other = boxes[::2].clone()
    a, b = boxes.cuda(), other.cuda()
    tensors = {"input_boxes": boxes, "input_other": other}
    for mode in ("IoU", "IoF", "IoS"):
        tensors["overlap/" + mode] = ops.gpu_iou_overlap(a, b, mode).cpu()
    scores = torch.linspace(1., .1, len(a), device="cpu").cuda()
    tensors["input_scores"] = scores.cpu()
    dets = torch.cat((a, scores[:, None]), dim=1)
    tensors["nms/legacy"] = ops.naive_nms(dets, .5).cpu()
    exclusive = a.clone()
    exclusive[:, 2:4] = torch.maximum(exclusive[:, 2:4], exclusive[:, :2] + .01)
    tensors["input_exclusive_boxes"] = exclusive.cpu()
    tensors["nms/torchvision"] = ops.torchvision_nms(exclusive, scores, .5).cpu()
    torch.save(tensors, args.output / "tensors.pt")
    return {"tested": list(tensors), "input": "Identical CPU-generated boxes, including subpixel and degenerate cases"}


def training_probe(args):
    import torch
    helpers = sibling("validate_training_accuracy")
    helpers.prepare_legacy_environment()
    sys.path.insert(0, str(args.source.resolve()))
    import up
    from up.accelerator import configure_accelerator, register_op
    configure_accelerator(backend=args.backend)
    if args.portable_ops:
        from up.accelerator.backends import portable
        for name in ("naive_nms", "gpu_iou_overlap", "torchvision_nms"):
            register_op(args.backend, name, getattr(portable, name), backward=False, replace=True)
    up.initialize()
    from up.runner.base_runner import BaseRunner
    from up.utils.general.global_flag import DIST_BACKEND
    DIST_BACKEND.backend = "dist"
    epochs = {"yolov5s": 300, "detr": 12, "vit_b16": 20}[args.model]
    config, _ = helpers.effective_config(args.source, json.loads(args.data.read_text()),
                                          args.model, args.seed, epochs, "multiseed_v2")
    config["runtime"]["accelerator"]["backend"] = args.backend
    if args.numerical_profile != "native":
        config["runtime"]["accelerator"]["numerics"] = args.numerical_profile
    if args.disable_convolution_library:
        config["args"]["nocudnn"] = True
    config["saver"].update(save_dir=str(args.output / "checkpoints"),
                            results_dir=str(args.output / "predictions"))
    runner = BaseRunner(config, work_dir=str(args.output), training=True)
    if args.kind == "inference":
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        runner.model.load_state_dict(state, strict=True)
        return {"checkpoint_sha256": helpers.file_hash(args.checkpoint),
                "model_state_sha256": helpers.state_hash(runner.model.state_dict()),
                "metrics": dict(runner.evaluate())}
    result = {"initial_state": helpers.state_hash(runner.model.state_dict()),
              "initial_metrics": dict(runner.evaluate()), "steps": [],
              "optimizer": {"type": type(runner.optimizer).__name__,
                            "groups": [{k: v for k, v in group.items() if k != "params"}
                                       for group in runner.optimizer.param_groups]}}
    tensors = {}
    step = -1
    random_calls = []

    def snapshot(prefix, value):
        if isinstance(value, torch.Tensor):
            if value.numel():
                # Samples are explicitly labelled; hashes cover the complete tensor.
                t = value.detach().cpu().contiguous()
                tensors[prefix] = t.flatten()[:2048].clone()
                return {"shape": list(t.shape), "dtype": str(t.dtype),
                        "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest(),
                        "mean": float(t.double().mean()), "norm": float(t.double().norm())}
        elif isinstance(value, dict):
            return {str(k): snapshot(prefix + "/" + str(k), v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [snapshot(prefix + "/" + str(i), v) for i, v in enumerate(value)]
        return value if isinstance(value, (str, int, float, bool, type(None))) else str(type(value))

    stages = {}
    stage_inputs = {}

    def hook(name):
        def capture(module, inputs, output):
            if step == args.capture_step:
                stages[name] = snapshot("forward/" + name, output)
                if name.endswith("body.conv1") or name == "backbone.embedding":
                    stage_inputs[name] = snapshot("input/" + name, inputs)
        return capture

    leaves = [(n, m) for n, m in runner.model.named_modules() if not list(m.children())]
    selected = dict(leaves[:40])
    selected.update((n, m) for n, m in leaves if n.endswith(("body.conv1", "body.bn1")))
    if args.capture_all:
        selected = {n: m for n, m in runner.model.named_modules()
                    if not list(m.children()) or isinstance(m, torch.nn.MultiheadAttention)}
    handles = [m.register_forward_hook(hook(n)) for n, m in selected.items()]
    import torch.nn.functional as F
    original_dropout, original_rand = F.dropout, torch.rand
    paired_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1000000)

    def dropout(input, p=.5, training=True, inplace=False):
        output = original_dropout(input, p, training, inplace)
        if args.paired_masks and training and 0 < p < 1:
            if inplace:
                raise ValueError("Diagnostic mask control does not support inplace dropout")
            mask = original_rand(input.shape, generator=paired_generator, device="cpu") >= p
            output = input * mask.to(device=input.device, dtype=input.dtype) / (1 - p)
        if step == args.capture_step and training and p > 0:
            name = "random/dropout/" + str(len(random_calls))
            random_calls.append({"kind": "dropout", "p": p,
                                 "input_zeros": int((input == 0).sum()),
                                 "observed_zero_mask": snapshot(name, output == 0)})
        return output

    def rand(*values, **kwargs):
        output = original_rand(*values, **kwargs)
        if step == args.capture_step:
            name = "random/rand/" + str(len(random_calls))
            random_calls.append({"kind": "rand", "values": snapshot(name, output)})
        return output

    F.dropout, torch.rand = dropout, rand
    runner._hooks("before_train")
    runner.model.train()
    dispatch_trace = None
    try:
        for step in range(args.steps):
            row = {"step": step, "rng_before_data": rng_state()}
            batch = runner.get_batch("train")
            row["batch_sha256"] = helpers.batch_hash(batch)
            row["rng_before_forward"] = rng_state()
            if args.trace_dispatch and step == args.capture_step:
                from trace_gpu_dispatch import DispatchTrace
                dispatch_trace = DispatchTrace()
                dispatch_trace.__enter__()
            loss = runner.forward_train(batch)
            row["loss"] = float(loss.detach())
            row["rng_after_forward"] = rng_state()
            if step == args.capture_step:
                row["stages"] = dict(stages)
                row["stage_inputs"] = dict(stage_inputs)
                row["loss_components"] = snapshot("loss", {
                    k: v for k, v in runner._temporaries["last_output"].items() if "loss" in k})
                row["random_calls"] = list(random_calls)
            if dispatch_trace is not None:
                dispatch_trace.phase = "backward"
            runner.backward(loss)
            if step == args.capture_step:
                row["gradients"] = {n: snapshot("grad/" + n, p.grad)
                                    for n, p in runner.model.named_parameters() if p.grad is not None}
            row["rng_after_backward"] = rng_state()
            if dispatch_trace is not None:
                dispatch_trace.phase = "update"
            runner.update()
            row["state_after_update"] = helpers.state_hash(runner.model.state_dict())
            runner.lr_scheduler.step()
            row["rng_after_update"] = rng_state()
            result["steps"].append(row)
            if dispatch_trace is not None:
                dispatch_trace.__exit__(None, None, None)
                (args.output / "dispatch-trace.json").write_text(
                    json.dumps(dispatch_trace.rows, indent=2, sort_keys=True))
                dispatch_trace = None
    finally:
        if dispatch_trace is not None:
            dispatch_trace.__exit__(None, None, None)
        F.dropout, torch.rand = original_dropout, original_rand
        for handle in handles:
            handle.remove()
    torch.save(tensors, args.output / "tensors.pt")
    return result


def compare(args):
    import torch
    a, b = args.compare
    reports = [json.loads((p / "report.json").read_text()) for p in (a, b)]
    values = [torch.load(p / "tensors.pt", weights_only=True, map_location="cpu") for p in (a, b)]
    comparisons = {}
    if values[0].keys() != values[1].keys() and not args.allow_schema_difference:
        raise ValueError("Different captured tensor keys")
    for key, x in values[0].items():
        if key not in values[1]:
            continue
        y = values[1][key]
        if x.shape != y.shape or x.dtype != y.dtype:
            comparisons[key] = {"contract_mismatch": True}
            continue
        comparisons[key] = {"equal": torch.equal(x, y),
                            "max_abs": float((x.double() - y.double()).abs().max()),
                            "unequal_elements": int((x != y).sum()), "elements": x.numel()}
    result = {"scope": "diagnostic only; tensor values are samples for training, complete for RNG",
              "left": str(a), "right": str(b), "tensors": comparisons,
              "captured_key_gaps": {"left_only": sorted(values[0].keys() - values[1].keys()),
                                    "right_only": sorted(values[1].keys() - values[0].keys())},
              "reports_sha256": [sibling("validate_training_accuracy").file_hash(p / "report.json")
                                 for p in (a, b)]}
    if "steps" in reports[0]:
        result["initial_state_equal"] = reports[0]["initial_state"] == reports[1]["initial_state"]
        result["steps"] = [{"step": x["step"], "losses": [x["loss"], y["loss"]],
                            **{key + "_equal": x[key] == y[key] for key in (
                                "batch_sha256", "rng_before_data", "rng_before_forward",
                                "rng_after_forward", "rng_after_backward", "state_after_update",
                                "rng_after_update")}}
                           for x, y in zip(reports[0]["steps"], reports[1]["steps"], strict=True)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("rng", "ops", "training", "inference"), default="rng")
    parser.add_argument("--backend", choices=("nvidia", "metax"), default="nvidia")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model", choices=("yolov5s", "detr", "vit_b16"))
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--capture-step", type=int, default=0)
    parser.add_argument("--capture-all", action="store_true")
    parser.add_argument("--trace-dispatch", action="store_true",
                        help="Read-only full-tensor hashes for the captured diagnostic step; slow")
    parser.add_argument("--portable-ops", action="store_true", help="Diagnostic provider ablation only")
    parser.add_argument("--paired-masks", action="store_true",
                        help="Diagnostic only: private CPU-generated masks, GPU forward/backward; NOT production fallback")
    parser.add_argument("--rng-wide", action="store_true", help="Also test the ViT MLP dropout shape on all-one input")
    parser.add_argument("--disable-convolution-library", action="store_true")
    parser.add_argument("--numerical-profile", choices=("native", "portable_v1"), default="native")
    parser.add_argument("--compare", type=Path, nargs=2)
    parser.add_argument("--allow-schema-difference", action="store_true",
                        help="Explicitly compare only common observations and report every missing key")
    parser.add_argument("--checkpoint", type=Path, help="Model-only tensor state for inference")
    parser.add_argument("--export-model", action="store_true",
                        help="Export model-only state from our own trusted full comparison checkpoint; CPU only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.export_model:
        if not args.checkpoint or args.output.exists():
            parser.error("Model export requires --checkpoint and a new output file")
        import torch
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state["model"], args.output)
        return
    if args.compare:
        compare(args)
        return
    if args.kind in ("training", "inference") and (not all((args.source, args.data, args.model)) or args.steps <= 0):
        parser.error("Training requires source, data, model and positive steps")
    if args.kind == "training" and not 0 <= args.capture_step < args.steps:
        parser.error("Capture step must be inside the diagnostic trajectory")
    if args.kind == "inference" and not args.checkpoint:
        parser.error("Inference requires a model-only --checkpoint")
    if args.kind == "ops" and not args.source:
        parser.error("Operator probe requires --source")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh diagnostic directory")
    args.output.mkdir(parents=True, exist_ok=True)
    configure()
    if args.kind == "rng" and args.numerical_profile != "native":
        if not args.source:
            parser.error("Portable GPU RNG requires --source")
        sys.path.insert(0, str(args.source.resolve()))
        from up.accelerator import configure_accelerator
        configure_accelerator({"runtime": {"accelerator": {
            "backend": args.backend, "numerics": args.numerical_profile}}})
    result = (rng_probe(args) if args.kind == "rng" else
              operator_probe(args) if args.kind == "ops" else training_probe(args))
    import torch
    result.update(status="completed", kind=args.kind, backend=args.backend, model=args.model,
                  seed=args.seed, portable_ops=args.portable_ops, paired_masks=args.paired_masks, torch=torch.__version__,
                  numerical_profile=args.numerical_profile,
                  numerics=(sys.modules["up.accelerator"].get_accelerator().numerics_state()
                            if "up.accelerator" in sys.modules else {"profile": "not_configured"}),
                  tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="diagnostic_only; not a completed training or accuracy acceptance")
    (args.output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
