"""Export/check the complete hardware boundary without loading a vendor SDK."""
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

NATIVE_RETURNS = {
    "roi_align": ("Tensor[R,C,pooled_h,pooled_w]", "features"),
    "psroi_align": ("Tensor[R,output_dim,group_size,group_size]", "features"),
    "psroi_pool": ("Tensor[R,output_dim,group_size,group_size]", "features"),
    "deformable_conv": ("NCHW Tensor", "input, offset, weight"),
    "sigmoid_focal_loss": ("Tensor[N,C] for reduction=none, otherwise Tensor[1]", "preds"),
    "softmax_focal_loss": ("Tensor[1]", "preds"),
    "cross_focal_loss": ("Tensor[N,C] for reduction=none, otherwise Tensor[1]", "preds"),
    "naive_nms": ("same-device int64 Tensor[K] of kept indices; input already score-sorted", "none"),
    "softer_nms": ("(modified_dets[K,9], keep_indices); preserves the legacy variance/score columns", "none"),
    "gpu_iou_overlap": ("Tensor[N,M]; legacy empty input returns Tensor[0]", "none"),
    "boxes_bev_iou_cpu": ("CPU Tensor[N,M] or ndarray matching the original input type", "none"),
    "boxes_iou_bev": ("same-device Tensor[N,M]", "none"),
    "boxes_iou3d_gpu": ("same-device Tensor[N,M]; no complete native BEV backward contract", "none"),
    "nms_gpu": ("(same-device int64 Tensor[K], None)", "none"),
    "nms_normal_gpu": ("(same-device int64 Tensor[K], None)", "none"),
    "points_in_boxes_cpu": ("CPU int32 Tensor[M,N] membership mask", "none"),
    "points_in_boxes_gpu": ("same-device int32 Tensor[B,N]; box index or -1", "none"),
    "roiaware_pool3d": ("Tensor[R,out_x,out_y,out_z,C]", "pts_feature"),
    "roipoint_pool3d": ("(Tensor[B,M,S,3+C], int32 Tensor[B,M] empty flags)", "none; upstream backward raises"),
    "rotate_iou_eval": ("numpy float32 ndarray[N,K]; complete evaluation boundary", "none"),
}

EXTRA = {
    "torchvision_nms": ("boxes, scores, iou_threshold", "vision", "same-device int64 Tensor[K]", "none"),
    "sync_batch_norm": ("input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size",
                        "normalization", "Tensor with the input shape", "input, weight, bias"),
    "sync_batch_norm_forward": ("module, value", "normalization", "Tensor with the input shape", "input and module affine parameters"),
    "link_sync_batch_norm": ("*args, **kwargs", "optional_linklink", "link.nn.SyncBNFunc.apply result", "supplier-owned complete autograd"),
    "build_fused_optimizer": ("config", "optional_optimizer", "supplier optimizer with unchanged parameter groups", "not applicable"),
    "reload_fused_optimizer": ("optimizer", "optional_optimizer", "None; rebinds/promotes optimizer master parameters in place", "not applicable"),
    "register_float_modules": ("modules, initialize=False", "optional_quantization", "None; supplier FP16 registration side effects", "not applicable"),
    "quantization_backend_types": ("", "optional_quantization", "dict of platform names to supplier BackendType values", "not applicable"),
}

DISTRIBUTED = {
    "init_process_group": "backend=None, *args, **kwargs",
    "new_group": "ranks=None, timeout=None, backend=None, **kwargs",
    "all_reduce": "tensor, op=SUM, group=None, async_op=False",
    "broadcast": "tensor, src, group=None, async_op=False",
    "all_gather": "tensor_list, tensor, group=None, async_op=False",
    "gather": "tensor, gather_list=None, dst=0, group=None, async_op=False",
    "barrier": "group=None, async_op=False, device_ids=None",
    "reduce": "tensor, dst, op=SUM, group=None, async_op=False",
    "reduce_scatter": "output, input_list, op=SUM, group=None, async_op=False",
    "all_gather_into_tensor": "output_tensor, input_tensor, group=None, async_op=False",
    "broadcast_object_list": "object_list, src=0, group=None, device=None, **kwargs",
    "all_gather_object": "object_list, obj, group=None",
    "destroy_process_group": "group=None",
    "is_available": "",
    "is_initialized": "",
    "get_rank": "group=None",
    "get_world_size": "group=None",
    "get_backend": "group=None",
    "wrap_model": "model, device_ids=None, **kwargs",
    "collective_device": "group=None",
    "cpu_group": "provider='dist'",
}

SPARSE = {
    "SubMConv3d": ("*args, **kwargs", "supplier nn.Module; original constructor and checkpoint keys"),
    "SparseConv3d": ("*args, **kwargs", "supplier nn.Module; original constructor and checkpoint keys"),
    "SparseInverseConv3d": ("*args, **kwargs", "supplier nn.Module; original constructor and checkpoint keys"),
    "SparseConvTensor": ("*args, **kwargs", "supplier sparse tensor; original constructor semantics"),
    "features": ("value", "Tensor[active_sites,channels]"),
    "replace_feature": ("value, new_features", "sparse tensor with unchanged geometry and replaced features"),
    "to_dense": ("value", "dense Tensor in the original supplier layout"),
    "is_sparse_convolution": ("module", "bool"),
    "sequential_forward": ("sequence, value", "sparse or dense result after ordered module execution"),
}

LEGACY_LINK = ("allreduce", "broadcast", "allgather", "gather", "is_initialized",
               "get_rank", "get_world_size", "barrier", "allreduce_async", "new_group",
               "get_local_rank", "initialize", "synchronize", "get_backend", "finalize")


def ast_signature(path, attribute):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parts = attribute.split(".")
    nodes = tree.body
    for part in parts:
        if part == "apply":
            part = "forward"
        node = next(value for value in nodes if isinstance(value, (ast.FunctionDef, ast.ClassDef))
                    and value.name == part)
        nodes = getattr(node, "body", [])
    arguments = copy.deepcopy(node.args)
    if arguments.args and arguments.args[0].arg in ("self", "ctx"):
        arguments.args.pop(0)
    return ast.unparse(arguments)


def build_contract(root=ROOT):
    sys.path.insert(0, str(root))
    import up.accelerator as accelerator
    from up.accelerator import ops
    from up.accelerator.registry import OPERATIONS

    if Path(accelerator.__file__).resolve().parent != root.resolve() / "up" / "accelerator":
        raise RuntimeError("Contract exporter imported a different source tree")
    runtime = []
    for name, value in vars(accelerator.TorchAccelerator).items():
        if name.startswith("_") and name != "__init__":
            continue
        if isinstance(value, property):
            runtime.append({"name": name, "signature": name, "kind": "property"})
        elif inspect.isfunction(value):
            runtime.append({"name": name, "signature": name + str(inspect.signature(value)), "kind": "method"})
    operations = []
    for name in sorted({name for _, name in OPERATIONS}):
        if name in ops.NATIVE_OPERATIONS:
            module, attribute, _ = ops.NATIVE_OPERATIONS[name]
            source = "up/accelerator/backends/nvidia_ops/" + module + ".py"
            signature = ast_signature(root / source, attribute)
            returns, gradients = NATIVE_RETURNS[name]
            category = "native_semantic_operation"
            convention = "positional .apply arguments" if attribute.endswith(".apply") else "original Python function"
        elif name in EXTRA:
            signature, category, returns, gradients = EXTRA[name]
            source = "up/accelerator/layers.py" if "batch_norm" in name else "up/accelerator/ops.py"
            convention = "complete supplier operation; keep positional/keyword compatibility"
        elif name.startswith("distributed."):
            short = name.split(".", 1)[1]
            signature = DISTRIBUTED[short]
            category, gradients, source = "distributed", "DDP-owned gradients; collectives do not promise autograd", "up/accelerator/distributed.py"
            returns = {"wrap_model": "DistributedDataParallel-compatible model with original parameters",
                       "collective_device": "torch.device; CPU for Gloo, active device otherwise",
                       "cpu_group": "Gloo process group"}.get(
                           short, "same return, mutation and async Work semantics as the supported PyTorch version")
            convention = "PyTorch-compatible core signature; version-specific extra kwargs belong to the supplier"
        elif name.startswith("sparse."):
            signature, returns = SPARSE[name.split(".", 1)[1]]
            category, gradients, source = "sparse", "supplier autograd where applicable; preserve feature gradients", "up/accelerator/sparse.py"
            convention = "original supplier constructor or exact facade function"
        elif name.startswith("legacy_link.") and name.split(".", 1)[1] in LEGACY_LINK:
            signature, category, returns = "*args, **kwargs", "optional_linklink", "unchanged spring.linklink supplier return"
            gradients, source = "supplier-defined; no invented backward", "up/accelerator/backends/legacy.py"
            convention = "opaque legacy SDK signature; pin the SDK and delegate without reinterpretation"
        else:
            raise ValueError("Missing semantic documentation for registered operation " + name)
        if name == "torchvision_nms":
            convention = "three positional arguments; provider keyword names are not a portable contract"
        owners = {owner: {"backward": entry["backward"], "dtypes": entry["dtypes"],
                          "variants": entry["variants"]}
                  for (owner, op_name), entry in sorted(OPERATIONS.items()) if op_name == name}
        operations.append({"name": name, "signature": "{}({})".format(name, signature),
                           "category": category, "returns": returns, "gradients": gradients,
                           "calling_convention": convention, "reference_source": source,
                           "requirement": "model_or_feature_dependent", "registered_providers": owners})
    entry_points = []
    for module, names in ((accelerator, ("configure_accelerator", "get_accelerator", "capabilities",
                                         "prepare_args", "activate_args", "register_backend", "register_op")),
                          (ops, ("call",))):
        for name in names:
            value = getattr(module, name)
            entry_points.append({"name": module.__name__ + "." + name,
                                 "signature": name + str(inspect.signature(value))})
    return {"contract_snapshot_version": 1, "runtime_class": "up.accelerator.TorchAccelerator",
            "runtime_attributes": ["name", "device_type", "communication_backend", "op_fallbacks", "extension_modules", "numerical_profiles"],
            "entry_points": entry_points, "runtime": runtime, "operations": operations,
            "status": "interface and registration snapshot, not hardware validation",
            "dtype_policy": "null metadata means unspecified, not all dtypes supported"}


def markdown(contract):
    lines = ["# Complete Hardware API Index", "", "Generated by `tools/export_accelerator_contract.py --write`.",
             "Read [INTERFACE.md](INTERFACE.md) for semantics and implementation obligations.", "",
             "Registration does not prove SDK availability or hardware validation. Null dtype/variant metadata is unspecified.",
             "", "## Entry Points", "", "| API | Signature |", "| --- | --- |"]
    for item in contract["entry_points"]:
        lines.append("| `{}` | `{}` |".format(item["name"], item["signature"]))
    lines += ["", "## Runtime", "", "Inherit `TorchAccelerator`; override incompatible methods only.", "",
              "| Member | Kind | Signature |", "| --- | --- | --- |"]
    for item in contract["runtime"]:
        lines.append("| `{}` | {} | `{}` |".format(item["name"], item["kind"], item["signature"]))
    lines += ["", "## Routed Operations", "", "| Operation | Signature | Result | Gradient Contract | Registered Owners |",
              "| --- | --- | --- | --- | --- |"]
    for item in contract["operations"]:
        lines.append("| `{}` | `{}` | {} | {} | {} |".format(
            item["name"], item["signature"], item["returns"], item["gradients"],
            ", ".join("`" + owner + "`" for owner in item["registered_providers"])))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.source.resolve()
    contract = build_contract(root)
    files = {root / "docs/hardware/interface_contract.json": json.dumps(contract, indent=2, sort_keys=True) + "\n",
             root / "docs/hardware/API_INDEX.md": markdown(contract)}
    if args.write:
        for path, content in files.items():
            path.write_text(content, encoding="utf-8", newline="\n")
    else:
        stale = [str(path.relative_to(root)) for path, content in files.items()
                 if not path.exists() or path.read_text(encoding="utf-8") != content]
        if stale:
            print("Stale/missing contract files: " + ", ".join(stale))
            return 1
    print("Hardware contract: {} runtime members, {} routed operations; snapshot matches source".format(
        len(contract["runtime"]), len(contract["operations"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
