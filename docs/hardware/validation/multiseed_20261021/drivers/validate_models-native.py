"""Bounded synthetic model, checkpoint, AMP and legacy-helper regression."""
import argparse
import ast
import copy
import hashlib
import importlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare_config(path):
    from up.utils.general.yaml_loader import load_yaml
    config = load_yaml(str(ROOT / path), "up")
    def update(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "group_size":
                    value[key] = None
                elif key == "min_kept":
                    value[key] = 128
                elif key == "base_anchors_file":
                    anchor = ("configs/det/yolov5/anchors.json" if "yolov5" in path else
                              "configs/det_3d/pointpillar/anchors.json")
                    value[key] = str(ROOT / anchor)
                elif key == "anchor_range":
                    value[key] = [0., -4., -3., 8., 4., 1.]
                elif key == "score_thresh":
                    value[key] = 0.
                else:
                    update(child)
        elif isinstance(value, list):
            for child in value:
                update(child)
    update(config["net"])
    return config


def image_input(kind, device):
    batch, height, width = (1, 128, 128) if kind == "faster_rcnn" else (2, 64, 128)
    if kind == "resnet18":
        height = width = 64
    elif kind == "vit_b16":
        height = width = 224
    elif kind == "yolov5s":
        height = width = 640
    value = {"image": torch.randn(batch, 3, height, width, device=device)}
    if kind in ("resnet18", "vit_b16"):
        value["gt"] = torch.tensor([1, 2], device=device)
    elif kind in ("faster_rcnn", "yolov5s"):
        value.update(
            image_info=[[height, width, 1., height, width, False] for _ in range(batch)],
            gt_bboxes=[torch.tensor([[15., 15., 65., 70., 1.]], device=device) for _ in range(batch)],
            gt_ignores=[torch.empty(0, 4, device=device) for _ in range(batch)])
    elif kind == "hrnet18":
        value["gt_semantic_seg"] = torch.randint(0, 19, (batch, height, width), device=device)
    return value


def point_input(config, device):
    channels = config["net"][0]["kwargs"]["num_point_features"]
    count, points_per_voxel = 64, 8
    coords = torch.zeros(count, 4, device=device, dtype=torch.int32)
    coords[:, 0] = torch.arange(count, device=device) // (count // 2)
    coords[:, 2] = torch.arange(count, device=device) // 8
    coords[:, 3] = torch.arange(count, device=device) % 8
    voxels = torch.rand(count, points_per_voxel, channels, device=device) * .1
    voxels[:, :, 0] += coords[:, 3:4] * .25 + .125
    voxels[:, :, 1] += coords[:, 2:3] * .25 - 4. + .125
    voxels[:, :, 2] -= 1.
    return {
        "voxels": voxels, "voxel_coords": coords,
        "voxel_num_points": torch.full((count,), points_per_voxel, dtype=torch.int32, device=device),
        "voxel_infos": {"grid_size": np.array([32, 32, 1]), "voxel_size": [.25, .25, 4.],
                        "point_cloud_range": [0., -4., -3., 8., 4., 1.]},
        "gt_boxes": torch.tensor([[[4., 0., -1., 3.9, 1.6, 1.56, 0., 1.]]] * 2, device=device),
        "class_names": ["Car", "Pedestrian", "Cyclist"], "batch_size": 2,
    }


def tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensors(child, prefix + "/" + key)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from tensors(child, prefix + "/" + str(index))


def legacy_helper(baseline):
    helper = importlib.import_module("up.models.model_helper")
    text = (Path(baseline) / "up/models/model_helper.py").read_text()
    node = next(node for node in ast.parse(text).body if isinstance(node, ast.ClassDef) and node.name == "ModelHelper")
    node.decorator_list = []
    namespace = dict(vars(helper))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<upstream ModelHelper>", "exec"), namespace)
    return namespace["ModelHelper"]


def run(args):
    from validate_training_accuracy import prepare_legacy_environment
    compatibility_shims = prepare_legacy_environment()
    from up.accelerator import configure_accelerator, get_accelerator
    configure_accelerator(backend="nvidia", reset=True)
    get_accelerator().set_device(args.device)
    import up
    up.initialize()
    from up.models.model_helper import ModelHelper
    from up.flags import ALIGNED_FLAG, DIST_BACKEND
    DIST_BACKEND.backend = "dist"
    torch.set_num_threads(2)
    legacy = legacy_helper(args.baseline) if args.baseline else None
    report = {
        "scope": "synthetic functional smoke; not dataset accuracy or convergence",
        "models": {}, "errors": {},
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version},
        "compatibility_shims": compatibility_shims,
        "synthetic_overrides": ["small input sizes", "SyncBN group_size=None",
                                "segmentation min_kept=128", "point cloud grid=32x32x1",
                                "absolute anchor file", "3D score threshold=0"],
    }
    configs = {
        "resnet18": "configs/cls/resnet/res18.yaml",
        "faster_rcnn": "configs/det/faster_rcnn/faster_rcnn_r50_fpn_1x.yaml",
        "pointpillars": "configs/det_3d/pointpillar/pointpillar.yaml",
        "hrnet18": "configs/seg/hrnet/hrnet18_1024x1024.yaml",
        "yolov5s": "configs/det/yolov5/yolov5_small_silu.yaml",
        "vit_b16": "configs/cls/vision_transformer/vit_b16_224.yaml",
    }
    for name, path in configs.items():
        if args.model and name != args.model:
            continue
        started = time.monotonic()
        try:
            torch.manual_seed(20261020)
            config = prepare_config(path)
            ALIGNED_FLAG.aligned = bool(config.get("runtime", {}).get("aligned", False))
            ALIGNED_FLAG.offset = 0 if ALIGNED_FLAG.aligned else 1
            model = ModelHelper(copy.deepcopy(config["net"])).to(get_accelerator().device).float()
            make_input = lambda: point_input(config, get_accelerator().device) if name == "pointpillars" else (
                image_input(name, get_accelerator().device))
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
            losses = []
            loss_components = []
            for _ in range(2):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                output = model(make_input())
                terms = [v.sum() for k, v in output.items() if "loss" in k and isinstance(v, torch.Tensor)]
                assert terms, "No training loss"
                loss = sum(terms)
                assert torch.isfinite(loss), loss
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                assert gradients and all(torch.isfinite(g).all() for g in gradients)
                optimizer.step()
                losses.append(float(loss.detach()))
                loss_components.append({key: float(value.sum().detach()) for key, value in output.items()
                                        if "loss" in key and isinstance(value, torch.Tensor)})
            model.eval()
            sample = make_input()
            with torch.no_grad():
                expected = dict(tensors(model(copy.deepcopy(sample))))
            assert expected and all(torch.isfinite(v).all() for v in expected.values())
            checkpoint = io.BytesIO()
            torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "optimizer": optimizer.state_dict()}, checkpoint)
            checkpoint.seek(0)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            restored = ModelHelper(copy.deepcopy(config["net"])).cuda(device=get_accelerator().device).float().eval()
            restored.load_state_dict(copy.deepcopy(state["model"]), strict=True)
            optimizer_restored = torch.optim.SGD(restored.parameters(), lr=1e-4)
            optimizer_restored.load_state_dict(state["optimizer"])
            with torch.no_grad():
                actual = dict(tensors(restored(copy.deepcopy(sample))))
            assert expected.keys() == actual.keys()
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], atol=1e-5, rtol=1e-4)
            legacy_ok = None
            if legacy:
                old = legacy(copy.deepcopy(config["net"])).cuda().float().eval()
                old.load_state_dict(copy.deepcopy(state["model"]), strict=True)
                with torch.no_grad():
                    old_outputs = dict(tensors(old(copy.deepcopy(sample))))
                assert expected.keys() == old_outputs.keys()
                for key in expected:
                    torch.testing.assert_close(expected[key], old_outputs[key], atol=1e-5, rtol=1e-4)
                legacy_ok = True
                del old
            entry = {
                "training_losses": losses, "finite_gradients": True, "inference": True,
                "loss_components": loss_components,
                "checkpoint_restore": True, "legacy_helper_state_and_forward": legacy_ok,
                "state_keys": len(state["model"]),
                "state_key_sha256": hashlib.sha256("\n".join(state["model"]).encode()).hexdigest(),
                "output_shapes": {key: list(value.shape) for key, value in expected.items()},
                "input_shape": list(sample["image"].shape) if "image" in sample else None,
                "class_count_config": config.get("num_classes"),
                "aligned": ALIGNED_FLAG.aligned,
                "seconds": round(time.monotonic() - started, 3),
            }
            if name == "resnet18":
                model.train()
                scaler = get_accelerator().grad_scaler(enabled=True)
                optimizer.zero_grad(set_to_none=True)
                with get_accelerator().autocast(enabled=True):
                    output = model(make_input())
                    loss = sum(v.sum() for k, v in output.items()
                               if "loss" in k and isinstance(v, torch.Tensor))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                assert torch.isfinite(loss)
                entry["amp_and_scaler"] = True
            report["models"][name] = entry
            print(name, "PASS", flush=True)
            del model, restored, optimizer, optimizer_restored, state, checkpoint
            get_accelerator().empty_cache()
        except Exception:
            report["errors"][name] = traceback.format_exc()
            print(name, report["errors"][name], flush=True)
    report["passed"] = not report["errors"]
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report["errors"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=["resnet18", "faster_rcnn", "pointpillars", "hrnet18", "yolov5s", "vit_b16"])
    run(parser.parse_args())
