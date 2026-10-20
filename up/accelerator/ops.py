"""Semantic operations; backend implementations own their complete autograd."""
from .registry import register_op


def call(name, *args, **kwargs):
    from . import get_accelerator
    return get_accelerator().call(name, *args, **kwargs)


def _entry(name):
    def dispatch(*args, **kwargs):
        return call(name, *args, **kwargs)
    dispatch.__name__ = name
    return dispatch


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
        ("iou3d_nms_utils", ("boxes_bev_iou_cpu", "boxes_iou_bev", "boxes_iou3d_gpu",
                             "nms_gpu", "nms_normal_gpu")),
        ("roiaware_pool3d_utils", ("points_in_boxes_cpu", "points_in_boxes_gpu"))):
    for _function in _functions:
        NATIVE_OPERATIONS[_function] = (_module, _function, False)
NATIVE_OPERATIONS["roiaware_pool3d"] = ("roiaware_pool3d_utils", "RoIAwarePool3dFunction.apply", True)
NATIVE_OPERATIONS["roipoint_pool3d"] = ("roipoint_pool3d_utils", "RoIPointPool3dFunction.apply", False)
NATIVE_OPERATIONS["rotate_iou_eval"] = ("rotate_iou", "rotate_iou_gpu_eval", False)


def _native(module, attribute):
    def implementation(*args, **kwargs):
        import importlib
        import torch
        if module in ("roi_align", "psroi_align", "psroi_pool", "deformable_conv") and args:
            if args[0].device.type == "cpu" and not torch.onnx.is_in_onnx_export():
                from .errors import CapabilityError
                raise CapabilityError("NVIDIA {} has no executable CPU implementation".format(module))
        value = importlib.import_module("up.accelerator.backends.nvidia_ops." + module)
        for part in attribute.split("."):
            value = getattr(value, part)
        return value(*args, **kwargs)
    return implementation


for _name, (_module, _attribute, _backward) in NATIVE_OPERATIONS.items():
    globals()[_name] = _entry(_name)
    register_op("nvidia", _name, _native(_module, _attribute), backward=_backward)

for _name in ("naive_nms", "gpu_iou_overlap"):
    register_op("portable", _name, "up.accelerator.backends.portable:" + _name, backward=False)
torchvision_nms = _entry("torchvision_nms")
register_op("portable", "torchvision_nms", "up.accelerator.backends.portable:torchvision_nms", backward=False)
register_op("nvidia", "torchvision_nms", "up.accelerator.backends.torch_backends:torchvision_nms", backward=False)
for _name in ("sync_batch_norm", "sync_batch_norm_forward"):
    globals()[_name] = _entry(_name)
    register_op("portable", _name, "up.accelerator.layers:" + _name, backward=True)


def _sparse_factory(name):
    def create(*args, **kwargs):
        from .sparse import _factory
        return _factory(name, *args, **kwargs)
    return create


for _name in ("SubMConv3d", "SparseConv3d", "SparseInverseConv3d", "SparseConvTensor"):
    register_op("portable", "sparse." + _name, _sparse_factory(_name))
for _name in ("sequential_forward", "features", "replace_feature", "to_dense", "is_sparse_convolution"):
    register_op("portable", "sparse." + _name, "up.accelerator.sparse:_" + _name)

for _name in ("build_fused_optimizer", "reload_fused_optimizer", "register_float_modules",
              "quantization_backend_types"):
    register_op("nvidia", _name, "up.accelerator.backends.legacy:" + _name)


def _link_entry(name):
    def invoke(*args, **kwargs):
        from .backends.legacy import link_call
        return link_call(name, *args, **kwargs)
    return invoke


for _name in ("allreduce", "broadcast", "allgather", "gather", "is_initialized",
              "get_rank", "get_world_size", "barrier", "allreduce_async", "new_group",
              "get_local_rank", "initialize", "synchronize", "get_backend", "finalize"):
    register_op("nvidia", "legacy_link." + _name, _link_entry(_name))
