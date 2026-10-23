"""Compare original UP Python/autograd bodies with explicit vendor bindings.

The reference is extracted from the pinned upstream, not the old hardware main.
Both paths use the same vendor-compiled, byte-identical upstream kernel source.
This isolates routing/autograd migration, not cross-vendor numerical equivalence.
"""
import argparse
import ast
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from up_platform import configure_platform
from up_platform.ops._native.loader import load_native


def original_module(root, name, extension, flags, psroi_layout_fix=False):
    path = root / "up/extensions/python" / (name + ".py")
    tree = ast.parse(path.read_text())
    if psroi_layout_fix and name == "psroi_pool":
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and ast.unparse(node) ==
                    "mapping_channel = mapping_channel.cuda()"):
                node.value = ast.Call(func=ast.Attribute(value=node.value, attr="contiguous", ctx=ast.Load()),
                                      args=[], keywords=[])
        ast.fix_missing_locations(tree)
    namespace = {"__name__": "upstream_reference_" + name, "logger": logging.getLogger(name), "ALIGNED_FLAG": flags}
    kept = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "ext":
            for entry in node.names:
                namespace[entry.asname or entry.name] = getattr(extension, entry.name)
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("up."):
            continue
        else:
            kept.append(node)
    for source_path, function in [("up/tasks/det_3d/data/data_utils.py", "check_numpy_to_torch"),
                                  ("up/tasks/det_3d/data/box_utils.py", "enlarge_box3d")]:
        helper = ast.parse((root / source_path).read_text())
        node = next(n for n in helper.body if isinstance(n, ast.FunctionDef) and n.name == function)
        namespace.update(np=np, torch=torch)
        exec(compile(ast.Module(body=[node], type_ignores=[]), source_path, "exec"), namespace)
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, np.ndarray):
        return [torch.as_tensor(value)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in tensors(child)]
    return []


def run(function, args, kwargs, grad_indices, device, training):
    inputs = [x.detach().clone().requires_grad_(training and i in grad_indices) if isinstance(x, torch.Tensor) else x
              for i, x in enumerate(args)]
    torch.manual_seed(131)
    device.manual_seed(131)
    with torch.set_grad_enabled(training):
        output = function(*inputs, **kwargs)
        floats = [x for x in tensors(output) if x.requires_grad]
        if training and grad_indices and floats:
            # Original PSRoI backward requires contiguous upstream gradients.
            torch.autograd.backward(floats, [torch.ones_like(x).contiguous() for x in floats])
    device.synchronize()
    gradients = [inputs[i].grad.detach().cpu() if inputs[i].grad is not None else None for i in grad_indices] if training else []
    return {"output": [x.detach().cpu() for x in tensors(output)], "gradient": gradients}


def compare(left, right):
    records = []
    for kind in ("output", "gradient"):
        if len(left[kind]) != len(right[kind]):
            return {"passed": False, "error": "different output/gradient count"}
        for a, b in zip(left[kind], right[kind]):
            if a is None or b is None:
                records.append({"passed": a is b, "kind": kind, "abs_max": None})
                continue
            same_shape = a.shape == b.shape and a.dtype == b.dtype
            passed = same_shape and (torch.allclose(a, b, rtol=1e-4, atol=1e-5) if a.is_floating_point() else torch.equal(a, b))
            error = float((a.double() - b.double()).abs().max()) if same_shape and a.numel() else 0.0
            records.append({"passed": bool(passed), "kind": kind, "abs_max": error,
                            "shape": list(a.shape), "dtype": str(a.dtype)})
    return {"passed": all(r["passed"] for r in records), "comparisons": records}


def cases(device):
    def rand(*shape):
        return torch.randn(shape, device=device)
    roi = torch.tensor([[0, 0.3, 0.4, 5.5, 5.7], [1, 1., 1., 4., 4.]], device=device)
    boxes = torch.tensor([[0., 0., 0., 2., 2., 2., 0.], [0.5, 0.2, 0., 1., 1., 1., 0.3], [4., 4., 0., 1., 1., 1., 0.]], device=device)
    pts = rand(16, 3).contiguous()
    scores = torch.tensor([0.9, 0.7, 0.3], device=device)
    dets = torch.tensor([[0, 0, 2, 2, .9], [.2, .2, 2.2, 2.2, .7], [4, 4, 6, 6, .3]], device=device)
    result = []
    def add(category, method, module, entry, args, gradients=(), kwargs=None):
        result.append((category, method, module, entry, args, gradients, kwargs or {}))
    for aligned in (False, True):
        add("detection", "roi_align", "roi_align", "RoIAlignFunction", [rand(2, 4, 8, 8), roi, 2, 2, 1., 2, "AVG"], (0,), {"aligned": aligned})
        add("detection", "roi_align", "roi_align", "RoIAlignFunction", [rand(2, 4, 8, 8), roi[:0], 2, 2, 1., 2, "AVG"], (0,), {"aligned": aligned})
    add("detection", "psroi_align", "psroi_align", "PSRoIAlignFunction", [rand(2, 8, 8, 8), roi, 2, 1., 2, 2], (0,))
    add("detection", "psroi_pool", "psroi_pool", "PSRoIPoolFunction", [rand(2, 8, 8, 8), roi, 2, 1., 2], (0,))
    add("deformable", "deform_conv", "deformable_conv", "DeformableConvFunction", [rand(1, 2, 5, 5), rand(1, 18, 5, 5) * .1, rand(2, 2, 3, 3), 2, (3, 3), (1, 1), (1, 1), (1, 1), 1, 1], (0, 1, 2))
    targets = torch.tensor([0, 1, 2, 3, 1, 0], dtype=torch.int32, device=device)
    focal = [rand(6, 3), targets, torch.tensor([3.], device=device), 2., .25, 3]
    for reduction in ("sum", "none"):
        add("focal", "sigmoid_focal_loss", "focal_loss", "SigmoidFocalLossFunction", focal + [reduction], (0,))
        add("focal", "cross_sigmoid_focal_loss", "cross_focal_loss", "CrossSigmoidFocalLossFunction", focal + [torch.tensor([0], dtype=torch.int32, device=device), reduction], (0,))
    add("focal", "softmax_focal_loss", "focal_loss", "SoftmaxFocalLossFunction", [rand(6, 4), targets, focal[2], 2., .25, 4], (0,))
    for offset in (0, 1):
        for value in (dets, dets.cpu()):
            add("detection", "naive_nms", "nms", "naive_nms", [value, .5], kwargs={"offset": offset})
    softer = torch.cat((dets[:, :4], torch.zeros(3, 4, device=device), dets[:, 4:]), 1)
    add("detection", "softer_nms", "nms", "softer_nms", [softer])
    for mode in ("IoU", "IoF", "IoS"):
        add("detection", "iou_overlap", "iou_overlap", "gpu_iou_overlap", [dets[:, :4], dets[:, :4], mode])
    add("detection", "iou_overlap", "iou_overlap", "gpu_iou_overlap", [dets[:0, :4], dets[:, :4]])
    add("pointcloud", "points_in_boxes_cpu", "roiaware_pool3d_utils", "points_in_boxes_cpu", [pts.cpu(), boxes.cpu()])
    add("pointcloud", "points_in_boxes_gpu", "roiaware_pool3d_utils", "points_in_boxes_gpu", [pts[None], boxes[None]])
    for mode in ("max", "avg"):
        add("pointcloud", "roiaware_pool3d", "roiaware_pool3d_utils", "RoIAwarePool3dFunction", [boxes, pts, rand(16, 3), 2, 16, mode], (2,))
    add("pointcloud", "roipoint_pool3d", "roipoint_pool3d_utils", "RoIPointPool3dFunction", [pts[None], rand(1, 16, 3), boxes[None], (0.1, 0.1, 0.1), 8])
    add("pointcloud", "boxes_bev_iou_cpu", "iou3d_nms_utils", "boxes_bev_iou_cpu", [boxes.cpu(), boxes.cpu()])
    add("pointcloud", "boxes_iou_bev", "iou3d_nms_utils", "boxes_iou_bev", [boxes, boxes])
    add("pointcloud", "boxes_iou3d", "iou3d_nms_utils", "boxes_iou3d_gpu", [boxes, boxes], (0, 1))
    add("pointcloud", "nms3d", "iou3d_nms_utils", "nms_gpu", [boxes, scores, .5])
    add("pointcloud", "nms3d_normal", "iou3d_nms_utils", "nms_normal_gpu", [boxes, scores, .5])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", choices=("nvidia", "metax"), required=True)
    parser.add_argument("--purpose", choices=("training", "inference"), required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-psroi-layout-fix", action="store_true",
                        help="Explicitly adapt only the reference PSRoIPool int workspace layout; retain the unmodified failure separately")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.pt').exists():
        raise FileExistsError('Keep previous trials; select a new output: '+str(args.output))
    context = configure_platform({"backend": args.vendor}, purpose=args.purpose)
    context.device.set_device(0)
    torch.manual_seed(131)
    extension = load_native(args.vendor)
    flags = SimpleNamespace(aligned=False, offset=1)
    references = {}
    records, raw = [], {}
    for index, (category, method, module, entry, inputs, gradients, kwargs) in enumerate(cases(context.device.device())):
        name = f"{category}.{method}/{index}"
        if module not in references:
            references[module] = original_module(args.baseline, module, extension, flags, args.reference_psroi_layout_fix)
        original = references[module][entry]
        original = original.apply if isinstance(original, type) else original
        flags.aligned, flags.offset = kwargs.get("aligned", False), kwargs.get("offset", 1)
        routed = getattr(context.category(category), method)
        phase = 'reference'
        try:
            left = run(original, inputs, {}, gradients, context.device, args.purpose == "training")
            phase = 'reference_repeat'
            repeat = run(original, inputs, {}, gradients, context.device, args.purpose == "training")
            phase = 'routed'
            right = run(routed, inputs, kwargs, gradients, context.device, args.purpose == "training")
            result = {"operation": name, "reference_repeat": compare(left, repeat), "routed": compare(left, right)}
            raw[name] = {"reference": left, "reference_repeat": repeat, "routed": right}
        except Exception as error:
            result = {"operation": name, "phase": phase, "error": repr(error)}
        records.append(result)
        print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(".pt")
    torch.save(raw, raw_path)
    passed = all(r.get("routed", {}).get("passed", False) for r in records)
    report = {"vendor": args.vendor, "purpose": args.purpose, "numerics": "native", "rtol": 1e-4, "atol": 1e-5,
              "scope": "same-vendor routing/autograd; shared upstream native source; not full acceptance",
              "reference_adaptations": ["PSRoIPool integer workspace .cuda().contiguous()"] if args.reference_psroi_layout_fix else [],
              "environment": context.device.environment(), "device": str(context.device.properties().name),
              "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(), "passed": passed, "records": records}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
