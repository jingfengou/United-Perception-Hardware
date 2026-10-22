"""Same-device DETR execution-path benchmark, separate from accuracy training.

Freeze actual UP batches once; the standalone process forbids importing UP.
Timing excludes tensor transfers, RNG restoration, numerical audits and files.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from benchmark_metax_detr import cache_manifest, distribution
from diagnose_detr_origin import cpu, digest, rng, restore_rng, setup, write_json
from validate_platform_equivalence import file_hash, source_manifest

MODES = ("official", "up_native", "up_v2")
ORDER = tuple((mode, repeat) for repeat in range(3)
              for mode in MODES[repeat:] + MODES[:repeat])
SEED, WARMUP, UPDATES, REPEATS = 131, 60, 120, 10


def initialize(source, config, purpose, profile):
    from validate_training_accuracy import prepare_legacy_environment
    prepare_legacy_environment()
    sys.path[:0] = [str(source / "packages/up-platform/src"), str(source)]
    import up
    from up.accelerator import configure_accelerator
    config = copy.deepcopy(config)
    config["runtime"]["accelerator"] = {"backend": "metax", "numerics": profile}
    context = configure_accelerator(config, purpose=purpose)
    up.initialize()
    setup()
    from up.flags import ALIGNED_FLAG, DIST_BACKEND
    DIST_BACKEND.backend = "dist"
    ALIGNED_FLAG.aligned = bool(config["runtime"].get("aligned", False))
    ALIGNED_FLAG.offset = 0 if ALIGNED_FLAG.aligned else 1
    return context, config


def prepare(args):
    import torch
    from prepare_cross_vendor_data import verify_data
    from validate_training_accuracy import effective_config
    args.output.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.data.read_text())
    verified = verify_data(data)
    config, _ = effective_config(args.source, data, "detr", SEED, 12, "multiseed_v2")
    context, config = initialize(args.source, config, "training", "native")
    from up.runner.base_runner import BaseRunner
    config["saver"].update(save_dir=str(args.output / "unused-checkpoints"),
                           results_dir=str(args.output / "unused-results"))
    runner = BaseRunner(config, work_dir=str(args.output), training=True)
    adapter = runner.model.detr
    core = adapter.detector
    names = {id(p): name for name, p in core.named_parameters()}
    initial = {"model": cpu(core.state_dict()), "up_model": cpu(runner.model.state_dict()),
               "criterion": cpu(adapter.criterion.state_dict()),
               "optimizer": cpu(runner.optimizer.state_dict()),
               "parameter_groups": [{**{k: v for k, v in group.items() if k != "params"},
                                     "params": [names[id(p)] for p in group["params"]]}
                                    for group in runner.optimizer.param_groups],
               "requires_grad": {name: p.requires_grad for name, p in core.named_parameters()},
               "detr": data["detr"], "seed": SEED, "config": config}
    torch.save(initial, args.output / "initial.pt")
    captured = {}

    def core_input(module, inputs):
        captured.update(images=cpu(inputs[0].tensors), mask=cpu(inputs[0].mask))

    def loss_input(module, inputs):
        captured["targets"] = cpu(inputs[1])

    handles = [core.register_forward_pre_hook(core_input),
               adapter.criterion.register_forward_pre_hook(loss_input)]
    index = {"training": [], "inference": []}
    with torch.no_grad():
        for phase, count in (("training", WARMUP), ("inference", 25)):
            runner.model.train(phase == "training")
            for number in range(count):
                batch = runner.get_batch("train" if phase == "training" else "test")
                captured.clear()
                state = rng()
                runner.model(dict(batch))
                packet = {"batch": cpu(dict(batch)), "rng": state, **captured}
                name = phase + "-{:03d}.pt".format(number)
                torch.save(packet, args.output / name)
                index[phase].append({"path": name, "sha256": file_hash(args.output / name),
                                     "input_sha256": digest({k: packet[k] for k in captured}),
                                     "shape": list(packet["images"].shape)})
    for handle in handles:
        handle.remove()
    assert digest(core.state_dict()) == digest(initial["model"])
    assert context.purpose == "training"
    assert sum(row["shape"][0] for row in index["inference"]) == 50
    write_json(args.output / "manifest.json", {"seed": SEED, "index": index,
               "data_sha256": file_hash(args.data), "verified_data": verified,
               "source": source_manifest(args.source), "initial_sha256": file_hash(args.output / "initial.pt"),
               "initial_model_sha256": digest(initial["model"]),
               "initial_optimizer_sha256": digest(initial["optimizer"]),
               "algorithm_config_sha256": digest(config)})


def compare(reference, actual):
    """Small JSON summary; full first observations are retained separately."""
    import torch
    if isinstance(reference, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or reference.shape != actual.shape or reference.dtype != actual.dtype:
            raise ValueError("Tensor metadata differs")
        if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
            raise ValueError("Nonfinite audit tensor")
        return {"exact": torch.equal(reference, actual),
                "outside_tolerance": int((~torch.isclose(actual, reference, rtol=1e-4, atol=1e-5)).sum()),
                "max_abs": float((reference.double() - actual.double()).abs().max()) if reference.numel() else 0.}
    if reference is None or actual is None:
        return {"exact": reference is actual, "outside_tolerance": int(reference is not actual), "max_abs": 0.}
    if isinstance(reference, dict):
        if reference.keys() != actual.keys():
            raise ValueError("Audit keys differ")
        parts = [compare(reference[k], actual[k]) for k in reference]
    elif isinstance(reference, (list, tuple)):
        if len(reference) != len(actual):
            raise ValueError("Audit lengths differ")
        parts = [compare(a, b) for a, b in zip(reference, actual, strict=True)]
    else:
        raise TypeError(type(reference))
    return {"exact": all(p["exact"] for p in parts),
            "outside_tolerance": sum(p["outside_tolerance"] for p in parts),
            "max_abs": max((p["max_abs"] for p in parts), default=0.)}


def run(args):
    import torch
    from diagnose_detr_origin import BlockUP, gradients, load_official, map_tensors
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    assert file_hash(args.fixtures / "initial.pt") == manifest["initial_sha256"]
    assert source_manifest(args.source) == manifest["source"]
    initial = torch.load(args.fixtures / "initial.pt", map_location="cpu", weights_only=False)
    context = None
    official = args.mode == "official"
    profile = "metax_deterministic_v2" if args.mode == "up_v2" else "native"
    if official:
        sys.meta_path.insert(0, BlockUP(platform=True))
        setup()
        core, criterion, nested = load_official(initial, args.criterion)
        core.to("cuda")
        criterion.to("cuda")
        from models.detr import PostProcess
        postprocess = PostProcess()
        model = core
    else:
        context, config = initialize(args.source, initial["config"], args.phase, profile)
        from up.models.model_helper import ModelHelper
        model = ModelHelper(copy.deepcopy(config["net"])).to(context.device)
        model.load_state_dict(initial["up_model"], strict=True)
        core, criterion = model.detr.detector, model.detr.criterion
    assert digest(core.state_dict()) == manifest["initial_model_sha256"]
    optimizer = None
    if args.phase == "training":
        parameters = dict(core.named_parameters())
        groups = [{**group, "params": [parameters[name] for name in group["params"]]}
                  for group in initial["parameter_groups"]]
        optimizer = torch.optim.AdamW(groups)
        optimizer.load_state_dict(initial["optimizer"])
        assert digest(optimizer.state_dict()) == manifest["initial_optimizer_sha256"]
    model.train(args.phase == "training")
    packets = []
    for row in manifest["index"][args.phase]:
        path = args.fixtures / row["path"]
        assert file_hash(path) == row["sha256"]
        packet = torch.load(path, map_location="cpu", weights_only=False)
        packet["gpu"] = map_tensors({k: v for k, v in packet.items() if k != "rng"}, lambda t: t.to("cuda"))
        packets.append(packet)

    def execute(packet, update=False, restore=True):
        data = packet["gpu"]
        if restore:
            restore_rng(packet["rng"])
        if official:
            prediction = core(nested(data["images"], data["mask"]))
            if args.phase == "training":
                terms = criterion(prediction, data["targets"])
                output = {name: value * criterion.weight_dict[name] for name, value in terms.items()
                          if name in criterion.weight_dict}
            else:
                from accuracy_detr_adapter import up_prediction_boxes
                info = data["batch"]["image_info"]
                sizes = data["images"].new_tensor([[float(x[0]), float(x[1])] for x in info])
                results = postprocess(prediction, sizes)
                rows = []
                offset = 0 if initial["config"]["runtime"].get("aligned", False) else 1
                for i, (result, meta) in enumerate(zip(results, info)):
                    boxes = up_prediction_boxes(result["boxes"], meta, offset)
                    rows.append(torch.cat((boxes.new_full((len(boxes), 1), i), boxes,
                                           result["scores"][:, None], (result["labels"] + 1).to(boxes.dtype)[:, None]), 1))
                output = {"dt_bboxes": torch.cat(rows)}
        else:
            output = model(dict(data["batch"]))
        if args.phase == "training":
            total = sum(value for name, value in output.items() if "loss" in name)
            model.zero_grad()
            total.backward()
            if update:
                torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad and p.grad is not None], .1, 2)
                optimizer.step()
            return total
        return output

    def reset():
        core.load_state_dict(initial["model"], strict=True)
        model.zero_grad()
        if optimizer is not None:
            optimizer.load_state_dict(copy.deepcopy(initial["optimizer"]))

    manager = torch.enable_grad if args.phase == "training" else torch.no_grad
    with manager():
        started = time.perf_counter()
        for packet in packets:
            execute(packet, update=optimizer is not None)
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - started
        reset()
        torch.cuda.synchronize()
        cache_before = cache_manifest(Path(os.environ["TRITON_CACHE_DIR"]))
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        events, walls, images = [], [], 0
        count = UPDATES if args.phase == "training" else len(packets) * 6
        losses = []
        for index in range(count):
            packet = packets[index % len(packets)]
            # Keep RNG restoration outside both timers, and identical in all modes.
            restore_rng(packet["rng"])
            torch.cuda.synchronize()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            begin.record()
            result = execute(packet, update=optimizer is not None, restore=False)
            end.record()
            torch.cuda.synchronize()
            walls.append(1000 * (time.perf_counter() - started))
            events.append(begin.elapsed_time(end))
            images += int(packet["images"].shape[0])
            if optimizer is not None:
                losses.append(result.detach())
        timing = {"wall_ms": distribution(walls), "stream_ms": distribution(events),
                  "images_per_second": 1000 * images / sum(walls), "images": images,
                  "baseline_allocated_bytes": baseline_allocated, "baseline_reserved_bytes": baseline_reserved,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "warmup_seconds": warmup_seconds}
        cache_after = cache_manifest(Path(os.environ["TRITON_CACHE_DIR"]))
        timing["cache_unchanged"] = cache_before == cache_after
        final = {"model": cpu(core.state_dict()), "optimizer": cpu(optimizer.state_dict()) if optimizer else None}
        torch.save(final, args.output / "final.pt")
        loss_values = [float(value) for value in losses]
        assert all(math.isfinite(value) for value in loss_values)
        assert all(torch.isfinite(value).all() for value in final["model"].values())
        audits = []
        indices = [0, 57, next(i for i, p in enumerate(packets) if p["images"].shape[0] == 1)] if optimizer else [0, 12, 24]
        reset()
        for case, index in enumerate(indices):
            observed = []
            handle = core.register_forward_hook(lambda module, inputs, result: observed.append(result))
            first = None
            comparisons = []
            for repeat in range(REPEATS):
                observed.clear()
                result = execute(packets[index])
                state = {"prediction": cpu(observed[0]),
                         "result": cpu(result if optimizer else result["dt_bboxes"]),
                         "gradient": cpu(gradients(core)) if optimizer else None}
                if first is None:
                    first = state
                    torch.save(state, args.output / ("audit-" + str(case) + ".pt"))
                comparisons.append({name: compare(value, state[name]) for name, value in first.items()})
            handle.remove()
            audits.append({"fixture_index": index, "repeats": REPEATS,
                           "reference_sha256": digest(first), "comparisons": comparisons})
    imported_up = any(name == "up" or name.startswith("up.") for name in sys.modules)
    imported_platform = any(name == "up_platform" or name.startswith("up_platform.") for name in sys.modules)
    if official:
        assert not imported_up and not imported_platform
    else:
        assert context.purpose == args.phase
    write_json(args.output / "report.json", {"status": "completed", "mode": args.mode, "phase": args.phase,
               "criterion": args.criterion if official else "aligned", "seed": SEED,
               "fixture_sha256": file_hash(args.fixtures / "manifest.json"), "source": manifest["source"]["sha256"],
               "initial_model": digest(initial["model"]), "initial_optimizer": digest(initial["optimizer"]),
               "timing": timing, "losses": loss_values, "audits": audits,
               "final_model": digest(final["model"]), "final_optimizer": digest(final["optimizer"]),
               "up_imported": imported_up, "platform_imported": imported_platform,
               "numerics": context.numerics_state() if context else {"profile": "native", "operations": {}},
               "rtol": 1e-4, "atol": 1e-5, "torch": torch.__version__,
               "python": sys.version, "torchvision": __import__("torchvision").__version__,
               "gpu": torch.cuda.get_device_name(0), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
               "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
               "driver_sha256": file_hash(Path(__file__)), "cache": cache_manifest(Path(os.environ["TRITON_CACHE_DIR"]))})


def suite(args):
    import fcntl
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output.parent / ("metax-device-" + args.device + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                  "pid": os.getpid(), "device": args.device, "jobs": []}
        try:
            for phase in ("inference", "training"):
                for mode, repeat in ORDER:
                    name = phase + "-" + mode + "-" + str(repeat)
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.device, OMP_NUM_THREADS="2",
                                       MKL_NUM_THREADS="2", PYTHONHASHSEED="0", DEFAULT_TASKS="cls:det",
                                       USER=os.environ.get("USER", "up-validation"), LOGNAME=os.environ.get("LOGNAME", "up-validation"),
                                       CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTORCH_DISABLE_CUDA_CUDNN_TF32="1",
                                       TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0", TRITON_CACHE_DIR=str(args.output / "triton-cache"))
                    command = [sys.executable, "-u", str(Path(__file__).resolve()), "run", "--source", str(args.source),
                               "--fixtures", str(args.fixtures), "--output", str(args.output / name),
                               "--mode", mode, "--phase", phase]
                    status["current"] = name
                    write_json(args.output / "execution.json", status)
                    with (args.output / (name + ".log")).open("xb") as stream:
                        code = subprocess.run(command, env=environment, cwd=args.source,
                                              stdout=stream, stderr=subprocess.STDOUT).returncode
                    status["jobs"].append({"name": name, "command": command, "exit_code": code})
                    if code:
                        raise RuntimeError(name + " failed; retain this attempt")
            status.update(status="completed", current=None)
        except Exception:
            status.update(status="failed", error=traceback.format_exc())
            raise
        finally:
            status["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(args.output / "execution.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "suite"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--mode", choices=MODES, default="official")
    parser.add_argument("--phase", choices=("training", "inference"), default="inference")
    parser.add_argument("--criterion", choices=("official", "aligned"), default="aligned")
    parser.add_argument("--device", default="2")
    args = parser.parse_args()
    if args.action == "prepare" and not args.data or args.action != "prepare" and not args.fixtures:
        parser.error("prepare needs --data; run/suite need --fixtures")
    globals()[args.action](args)


if __name__ == "__main__":
    main()
