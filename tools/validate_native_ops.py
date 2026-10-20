"""Run upstream or routed operations in separate processes and compare artifacts."""
import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def baseline_modules(root):
    """Isolate upstream operator imports from unrelated application dependencies."""
    import up.flags as flags
    for name in ("up.utils", "up.utils.general", "up.tasks", "up.tasks.det_3d",
                 "up.tasks.det_3d.data", "up_validation", "up_validation.python"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    sys.modules["up.utils.general.global_flag"] = flags
    logger = types.ModuleType("up.utils.general.log_helper")
    logger.default_logger = logging.getLogger("global")
    sys.modules[logger.__name__] = logger
    load_file("up.tasks.det_3d.data.data_utils", root / "up/tasks/det_3d/data/data_utils.py")
    geometry = types.ModuleType("up.tasks.det_3d.data.box_utils")
    geometry.check_numpy_to_torch = sys.modules["up.tasks.det_3d.data.data_utils"].check_numpy_to_torch
    text = (root / "up/tasks/det_3d/data/box_utils.py").read_text()
    function = next(node for node in ast.parse(text).body
                    if isinstance(node, ast.FunctionDef) and node.name == "enlarge_box3d")
    exec(ast.get_source_segment(text, function), geometry.__dict__)
    sys.modules[geometry.__name__] = geometry
    binary = next((root / "up/extensions").glob("_C*.so"))
    sys.modules["up_validation.ext"] = load_file("up_validation._C", binary)
    modules = {}
    for name in ("nms", "roi_align", "psroi_align", "psroi_pool", "deformable_conv",
                 "focal_loss", "cross_focal_loss", "iou_overlap", "iou3d_nms_utils",
                 "roiaware_pool3d_utils", "roipoint_pool3d_utils"):
        modules[name] = load_file("up_validation.python." + name,
                                  root / ("up/extensions/python/" + name + ".py"))
    return modules


def capture(function, gradient_inputs=()):
    value = function()
    outputs = value if isinstance(value, tuple) else (value,)
    differentiable = [output for output in outputs
                      if isinstance(output, torch.Tensor) and output.requires_grad]
    gradients = torch.autograd.grad(
        differentiable, gradient_inputs,
        grad_outputs=[torch.ones_like(output).contiguous() for output in differentiable],
        allow_unused=True) if (
        gradient_inputs and differentiable) else ()
    def cpu(item):
        return item.detach().cpu() if isinstance(item, torch.Tensor) else item
    torch.cuda.synchronize()
    return {"outputs": [cpu(item) for item in outputs], "gradients": [cpu(item) for item in gradients]}


def run(args):
    from up.accelerator import configure_accelerator, ops
    from up.flags import ALIGNED_FLAG
    configure_accelerator(backend="nvidia", reset=True)
    torch.cuda.set_device(args.device)
    modules = baseline_modules(Path(args.baseline)) if args.baseline else {}
    def invoke(module, function, op, *values):
        if modules:
            target = modules[module]
            for part in function.split("."):
                target = getattr(target, part)
            return target(*values)
        return ops.call(op, *values)

    result = {"cases": {}, "errors": {}, "environment": {
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version,
        "device": torch.cuda.get_device_name(), "mode": "upstream" if modules else "routed"}}
    def case(name, function):
        torch.manual_seed(20261020)
        try:
            result["cases"][name] = function()
        except Exception as exc:
            result["errors"][name] = type(exc).__name__ + ": " + str(exc)

    boxes = torch.tensor([[0., 0., 8., 8., .9], [1., 1., 7., 7., .8],
                          [20., 20., 24., 24., .7]], device="cuda")
    vision_root = Path(args.baseline) if modules else ROOT
    vision_wrapper = load_file(
        "up_validation_vision_wrapper", vision_root / "up/tasks/det/models/utils/nms_wrapper.py")
    case("torchvision_nms", lambda: capture(lambda: vision_wrapper.nms(boxes[[1, 0, 2]])))
    for offset in (0, 1):
        ALIGNED_FLAG.offset = offset
        case("nms_offset_" + str(offset),
             lambda: capture(lambda: invoke("nms", "naive_nms", "naive_nms", boxes, .5)))
    dets = torch.tensor([[0., 0., 8., 8., 1., 1., 1., 1., .9],
                         [1., 1., 7., 7., 1., 1., 1., 1., .8]], device="cuda")
    case("softer_nms", lambda: capture(lambda: invoke("nms", "softer_nms", "softer_nms", dets.clone())))
    for mode in ("IoU", "IoF", "IoS"):
        for dtype in (torch.float32, torch.float16):
            case("overlap_" + mode + "_" + str(dtype),
                 lambda mode=mode, dtype=dtype: capture(lambda: invoke(
                     "iou_overlap", "gpu_iou_overlap", "gpu_iou_overlap",
                     boxes[:, :4].to(dtype), boxes[:2, :4].to(dtype), mode)))
    case("overlap_empty", lambda: capture(lambda: invoke(
        "iou_overlap", "gpu_iou_overlap", "gpu_iou_overlap", boxes[:0], boxes)))
    rois = torch.tensor([[0., 1., 1., 5., 5.], [0., .5, .5, 4., 4.]], device="cuda")
    for aligned in (False, True):
        for dtype in (torch.float32, torch.float16):
            def roi(aligned=aligned, dtype=dtype):
                ALIGNED_FLAG.aligned = aligned
                feature = torch.randn(1, 2, 8, 8, device="cuda", dtype=dtype, requires_grad=True)
                return capture(lambda: invoke("roi_align", "RoIAlignFunction.apply", "roi_align",
                                               feature, rois.to(dtype), 2, 2, 1., 2, "AVG"), (feature,))
            case("roi_align_{}_{}".format(aligned, dtype), roi)
    ALIGNED_FLAG.aligned = False
    for module, function in (("psroi_align", "PSRoIAlignFunction.apply"), ("psroi_pool", "PSRoIPoolFunction.apply")):
        def pooling(module=module, function=function):
            feature = torch.randn(1, 8, 8, 8, device="cuda", requires_grad=True)
            values = (feature, rois, 2, 1., 2) + ((2,) if module == "psroi_align" else ())
            return capture(lambda: invoke(module, function, module, *values), (feature,))
        case(module, pooling)
    def deformable():
        value = torch.randn(1, 2, 5, 5, device="cuda", requires_grad=True)
        offset = (torch.randn(1, 18, 5, 5, device="cuda") * .1).requires_grad_()
        weight = torch.randn(2, 2, 3, 3, device="cuda", requires_grad=True)
        return capture(lambda: invoke("deformable_conv", "DeformableConvFunction.apply", "deformable_conv",
                                      value, offset, weight, 2, (3, 3), (1, 1), (1, 1), (1, 1), 1, 1),
                       (value, offset, weight))
    case("deformable_conv", deformable)
    for name, function in (
            ("sigmoid_focal_loss", "SigmoidFocalLossFunction.apply"),
            ("softmax_focal_loss", "SoftmaxFocalLossFunction.apply"),
            ("cross_focal_loss", "CrossSigmoidFocalLossFunction.apply")):
        def focal(name=name, function=function):
            value = torch.randn(4, 3, device="cuda", requires_grad=True)
            target = torch.tensor([0, 1, 2, -1], device="cuda", dtype=torch.int32)
            weight = torch.tensor([2.], device="cuda")
            extra = (torch.tensor([0], dtype=torch.int32, device="cuda"), "sum") if name == "cross_focal_loss" else (
                () if name == "softmax_focal_loss" else ("sum",))
            module = "cross_focal_loss" if name == "cross_focal_loss" else "focal_loss"
            return capture(lambda: invoke(module, function, name, value, target, weight, 2., .25, 3, *extra), (value,))
        case(name, focal)
    boxes3d = torch.tensor([[0., 0., 0., 2., 2., 2., 0.],
                            [.5, .5, 0., 2., 2., 2., .1],
                            [5., 5., 0., 2., 2., 2., 0.]], device="cuda")
    for name in ("boxes_iou_bev", "boxes_iou3d_gpu"):
        case(name, lambda name=name: capture(lambda: invoke(
            "iou3d_nms_utils", name, name, boxes3d, boxes3d[:2])))
    case("boxes_bev_iou_cpu", lambda: capture(lambda: invoke(
        "iou3d_nms_utils", "boxes_bev_iou_cpu", "boxes_bev_iou_cpu", boxes3d.cpu(), boxes3d[:2].cpu())))
    for name in ("nms_gpu", "nms_normal_gpu"):
        case(name, lambda name=name: capture(lambda: invoke(
            "iou3d_nms_utils", name, name, boxes3d, boxes[:, 4].contiguous(), .5)))
    points = torch.tensor([[0., 0., 0.], [.2, .2, .2], [.3, .1, 0.], [5., 5., 0.]], device="cuda")
    case("points_in_boxes_cpu", lambda: capture(lambda: invoke(
        "roiaware_pool3d_utils", "points_in_boxes_cpu", "points_in_boxes_cpu", points.cpu(), boxes3d.cpu())))
    case("points_in_boxes_gpu", lambda: capture(lambda: invoke(
        "roiaware_pool3d_utils", "points_in_boxes_gpu", "points_in_boxes_gpu",
        points.unsqueeze(0), boxes3d.unsqueeze(0))))
    for mode in ("max", "avg"):
        def aware(mode=mode):
            feature = torch.randn(4, 2, device="cuda", requires_grad=True)
            return capture(lambda: invoke("roiaware_pool3d_utils", "RoIAwarePool3dFunction.apply",
                                          "roiaware_pool3d", boxes3d, points, feature, 2, 8, mode), (feature,))
        case("roiaware_" + mode, aware)
    case("roipoint_pool3d", lambda: capture(lambda: invoke(
        "roipoint_pool3d_utils", "RoIPointPool3dFunction.apply", "roipoint_pool3d",
        points.unsqueeze(0), points[:, :2].contiguous().unsqueeze(0), boxes3d.unsqueeze(0), (0., 0., 0.), 4)))
    def rotate():
        array = np.array([[0., 0., 2., 2., 0.], [5., 5., 2., 2., .3]], dtype=np.float32)
        if modules:
            path = Path(args.baseline) / "up/tasks/det_3d/data/metrics/kitti_object_eval_python/rotate_iou.py"
            module = load_file("validation_rotate", path)
            value = module.rotate_iou_gpu_eval(array, array, device_id=args.device)
        else:
            value = ops.call("rotate_iou_eval", array, array, device_id=args.device)
        return {"outputs": [torch.from_numpy(value)],
                "gradients": []}
    case("rotate_iou_eval", rotate)
    torch.save(result, args.output)
    print(json.dumps({"cases": len(result["cases"]), "errors": result["errors"]}, indent=2))


def compare(args):
    original = torch.load(args.compare[0], map_location="cpu", weights_only=False)
    changed = torch.load(args.compare[1], map_location="cpu", weights_only=False)
    errors = {}
    for name, expected in original["cases"].items():
        try:
            assert name in changed["cases"], changed["errors"].get(name, "missing")
            for kind in ("outputs", "gradients"):
                actual = changed["cases"][name][kind]
                assert len(expected[kind]) == len(actual)
                for first, second in zip(expected[kind], actual):
                    if isinstance(first, torch.Tensor):
                        torch.testing.assert_close(first, second, rtol=1e-4, atol=1e-5)
                    else:
                        assert first == second
        except Exception as exc:
            errors[name] = str(exc)
    new_failures = set(changed["errors"]) - set(original["errors"])
    errors.update({name: changed["errors"][name] for name in new_failures})
    report = {"passed": not errors, "compared_cases": len(original["cases"]),
              "regressions": errors, "upstream_gaps": original["errors"],
              "environment": changed["environment"],
              "artifact_sha256": {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                  for path in args.compare}}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compare", nargs=2)
    arguments = parser.parse_args()
    compare(arguments) if arguments.compare else run(arguments)
