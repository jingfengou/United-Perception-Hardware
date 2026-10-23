"""Lazy catalog of the original 73 semantic operations, with explicit scopes."""
import json
from pathlib import Path

from .registry_factory import register_op

NVIDIA_MODULES = {
    "nms": "nms.nvidia.nms", "vision": "nms.nvidia.vision",
    "roi_align": "roi.nvidia.roi_align", "psroi_align": "roi.nvidia.psroi_align",
    "psroi_pool": "roi.nvidia.psroi_pool", "deformable_conv": "deform_conv.nvidia.deformable_conv",
    "focal_loss": "focal_loss.nvidia.focal_loss", "cross_focal_loss": "focal_loss.nvidia.cross_focal_loss",
    "iou_overlap": "iou.nvidia.iou_overlap", "iou3d_nms_utils": "iou.nvidia.iou3d_nms_utils",
    "rotate_iou": "iou.nvidia.rotate_iou", "roiaware_pool3d_utils": "pointcloud.nvidia.roiaware_pool3d_utils",
    "roipoint_pool3d_utils": "pointcloud.nvidia.roipoint_pool3d_utils",
}

NATIVE_OPERATIONS = {
    "roi_align": ("roi_align", "RoIAlignFunction.apply", True),
    "psroi_align": ("psroi_align", "PSRoIAlignFunction.apply", True),
    "psroi_pool": ("psroi_pool", "PSRoIPoolFunction.apply", True),
    "deformable_conv": ("deformable_conv", "DeformableConvFunction.apply", True),
    "sigmoid_focal_loss": ("focal_loss", "SigmoidFocalLossFunction.apply", True),
    "softmax_focal_loss": ("focal_loss", "SoftmaxFocalLossFunction.apply", True),
    "cross_focal_loss": ("cross_focal_loss", "CrossSigmoidFocalLossFunction.apply", True),
}
for _module, _functions in (
        ("nms", ("naive_nms", "softer_nms")),
        ("iou_overlap", ("gpu_iou_overlap",)),
        ("iou3d_nms_utils", ("boxes_bev_iou_cpu", "boxes_iou_bev", "boxes_iou3d_gpu", "nms_gpu", "nms_normal_gpu")),
        ("roiaware_pool3d_utils", ("points_in_boxes_cpu", "points_in_boxes_gpu"))):
    for _function in _functions:
        NATIVE_OPERATIONS[_function] = (_module, _function, False)
NATIVE_OPERATIONS["roiaware_pool3d"] = ("roiaware_pool3d_utils", "RoIAwarePool3dFunction.apply", True)
NATIVE_OPERATIONS["roipoint_pool3d"] = ("roipoint_pool3d_utils", "RoIPointPool3dFunction.apply", False)
NATIVE_OPERATIONS["rotate_iou_eval"] = ("rotate_iou", "rotate_iou_gpu_eval", False)

TRAINING_ONLY = frozenset(("sigmoid_focal_loss", "softmax_focal_loss", "cross_focal_loss",
                           "build_fused_optimizer", "reload_fused_optimizer", "register_float_modules",
                           "sync_batch_norm", "link_sync_batch_norm", "distributed.wrap_model"))


def native_operation(module, attribute):
    def implementation(*args, **kwargs):
        import importlib
        import torch
        from .errors import CapabilityError
        if module in ("roi_align", "psroi_align", "psroi_pool", "deformable_conv") and args:
            if args[0].device.type == "cpu" and not torch.onnx.is_in_onnx_export():
                raise CapabilityError("NVIDIA {} has no executable CPU implementation".format(module))
        value = importlib.import_module("up_platform.ops." + NVIDIA_MODULES[module])
        for part in attribute.split("."):
            value = getattr(value, part)
        if module == "roi_align":
            return value(*args, kwargs.pop("aligned", False), **kwargs)
        return value(*args, **kwargs)
    return implementation


def _sparse_factory(name):
    def create(*args, **kwargs):
        from .ops.sparse.common import _factory
        return _factory(name, *args, **kwargs)
    return create


def _link_entry(name):
    def invoke(*args, **kwargs):
        from .base.nvidia_legacy import link_call
        return link_call(name, *args, **kwargs)
    return invoke


def register_catalog():
    contract = json.loads((Path(__file__).parent / "common" / "legacy_contract.json").read_text(encoding="utf-8"))
    for operation in contract["operations"]:
        name = operation["name"]
        for owner, metadata in operation["registered_providers"].items():
            # The legacy manifest also records experimental providers. They are
            # supplied by the selected policy, not registered in the native catalog.
            if owner not in ("nvidia", "portable"):
                continue
            conventions = ("offset",) if name == "naive_nms" else (("aligned",) if name == "roi_align" else ())
            if owner == "nvidia" and name in NATIVE_OPERATIONS:
                module, attribute, _ = NATIVE_OPERATIONS[name]
                value = native_operation(module, attribute)
            elif name == "torchvision_nms" and owner == "nvidia":
                value = "up_platform.ops.nms.nvidia.vision:torchvision_nms"
            elif name in ("naive_nms", "torchvision_nms"):
                value = "up_platform.ops.nms.common:" + name
            elif name == "gpu_iou_overlap":
                value = "up_platform.ops.iou.common:" + name
            elif name in ("sync_batch_norm", "sync_batch_norm_forward"):
                value = "up_platform.ops.normalization.common:" + name
            elif name == "link_sync_batch_norm":
                value = "up_platform.ops.normalization.common:_link_sync_batch_norm"
            elif name.startswith("sparse."):
                short = name.split(".", 1)[1]
                value = (_sparse_factory(short) if short in ("SubMConv3d", "SparseConv3d", "SparseInverseConv3d", "SparseConvTensor")
                         else "up_platform.ops.sparse.common:_" + short)
            elif name == "distributed.wrap_model":
                value = "up_platform.training.runtime.distributed:wrap_model"
            elif name in ("distributed.collective_device", "distributed.cpu_group"):
                value = "up_platform.common.collectives:_" + name.split(".", 1)[1]
            elif name.startswith("distributed."):
                value = "torch.distributed:" + name.split(".", 1)[1]
            elif name.startswith("legacy_link."):
                value = _link_entry(name.split(".", 1)[1])
            else:
                value = "up_platform.base.nvidia_legacy:" + name
            dependencies = ("torch",)
            if owner == "nvidia" and name in NATIVE_OPERATIONS:
                dependencies = ("numba",) if name == "rotate_iou_eval" else ("up_platform.backends.nvidia._C",)
            elif owner == "nvidia" and name == "torchvision_nms":
                dependencies = ("torchvision.ops",)
            elif name.startswith("sparse."):
                dependencies = ("provider:sparse",)
            elif name.startswith("legacy_link.") or name in (
                    "link_sync_batch_norm", "build_fused_optimizer", "reload_fused_optimizer", "register_float_modules"):
                dependencies = ("provider:link",)
            elif name == "quantization_backend_types":
                dependencies = ("provider:quantization",)
            register_op(owner, name, value, scope="training" if name in TRAINING_ONLY else "common",
                        conventions=conventions, dependencies=dependencies, **metadata)
