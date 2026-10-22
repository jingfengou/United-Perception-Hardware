"""Declarative semantic contracts shared by documentation and conformance checks.

The frozen legacy manifest defines the ABI to migrate, not a hardware pass list.
New operations need a semantic record here as well as a registry entry.
"""
import ast
import copy
import inspect
import json
from pathlib import Path

from .catalog import NATIVE_OPERATIONS, TRAINING_ONLY
from .registry import OPERATIONS

ROOT = Path(__file__).parent

# Inputs, layout, empty-input behavior, mutation and export are explicit per operation.
NATIVE_SEMANTICS = {
    "roi_align": (
        "features[N,C,H,W]; rois[R,5]=batch,x1,y1,x2,y2; pooled_h/w; spatial_scale; sampling_ratio; AVG/MAX; aligned=False",
        "same device; NCHW features and contiguous RoIs of the feature dtype",
        "R=0 returns [0,C,pooled_h,pooled_w] and zero feature gradients",
        "saves features/RoIs and the aligned convention for complete backward; does not read global flags",
        "legacy RoiAlign node and original attributes; CPU forward is export-only, not a numeric fallback"),
    "psroi_align": (
        "features[N,output_dim*G*G,H,W]; rois[R,5]; group_size G; spatial_scale; output_dim; sampling_ratio",
        "same device; contiguous NCHW features and RoIs",
        "R=0 retains the original output and backward shapes",
        "saves channel mapping, RoIs and feature size for feature backward",
        "legacy PSRoiAlign node/attributes; original CPU export limitation remains"),
    "psroi_pool": (
        "features[N,output_dim*G*G,H,W]; rois[R,5]; group_size; spatial_scale; output_dim",
        "same device; contiguous NCHW features and RoIs",
        "R=0 retains original empty output and zero feature backward",
        "saves channel mapping, RoIs and input size; preserves discrete binning",
        "legacy PSRoiPool node/attributes; CPU numeric execution is not supported"),
    "deformable_conv": (
        "input NCHW; offset channels=2*deformable_groups*kH*kW; weight[O,I/groups,kH,kW]; stride/padding/dilation/groups",
        "same device/dtype and original contiguous layouts; preserve im2col_step constraints",
        "preserve supplier shape restrictions and exceptions; do not manufacture an empty output",
        "saves input, offset, weight, columns and ones for all three differentiable inputs",
        "legacy DeformConv symbol/attributes; no extra forward-only substitution"),
    "sigmoid_focal_loss": (
        "preds[N,C] logits; targets[N]; weight_pos normalizer; gamma; alpha; num_classes; reduction",
        "same accelerator; preserve native target dtype and background/ignore-label conventions",
        "N=0 follows native reduction and normalizer rules; no invented zero-loss shortcut",
        "saves logits, targets and normalization for preds backward; none returns [N,C], reduced returns [1]",
        "no guaranteed ONNX symbolic; export must fail explicitly if unsupported"),
    "softmax_focal_loss": (
        "preds[N,C]; targets[N]; weight_pos; gamma; alpha; num_classes; original class and ignore conventions",
        "same accelerator; native contiguous logits and label layout",
        "preserve native empty/reduction restrictions and normalization",
        "saves softmax intermediates and target state for preds backward; result stays [1]",
        "no guaranteed ONNX symbolic"),
    "cross_focal_loss": (
        "preds[N,C]; targets[N]; weight_pos; gamma; alpha; num_classes; gt_class_to_avoid; reduction",
        "same accelerator; retain label and exclusion-table layouts",
        "preserve native empty/reduction rules and exceptions",
        "saves complete exclusion/normalization state for preds backward; no reduction shape changes",
        "no guaranteed ONNX symbolic"),
    "naive_nms": (
        "dets[N,5]=x1,y1,x2,y2,score already sorted descending; thresh; explicit offset=0 (UP facade passes 0/1)",
        "same-device result; contiguous detections; no internal score reorder",
        "N=0 returns same-device int64[0]",
        "does not mutate detections; kept indices retain original index semantics; no backward",
        "no built-in generic ONNX NMS rewrite; keep original tracing/export behavior"),
    "softer_nms": (
        "dets[N,9] coordinates, variance and score columns; thresh/sigma/iou_thresh/iou_method/iou_sigma/method",
        "legacy explicit CPU internal phase; returns detections/indices to original device",
        "preserve native N=0 behavior and invalid-method exceptions; no new empty shortcut",
        "works on detached CPU detections and updates boxes/scores/variance by original rules",
        "no guaranteed export; existing CPU phase is not permission for other silent CPU fallbacks"),
    "gpu_iou_overlap": (
        "b1[N,4], b2[M,4]; mode IoU/IoF/IoS; inclusive coordinates and denominator clamp >=1",
        "same device/dtype; contiguous box matrices",
        "either input empty returns the legacy [0] tensor, not [N,M]",
        "no input mutation or full backward contract",
        "preserve original trace behavior; no dedicated symbolic guarantee"),
    "boxes_bev_iou_cpu": (
        "boxes_a[N,7], boxes_b[M,7]=x,y,z,dx,dy,dz,heading; heading in radians",
        "explicit CPU contiguous float32 Tensor or NumPy; return type follows original input",
        "retain native empty shape and restrictions; test before declaring support",
        "no mutation or gradient contract; explicit CPU API, not a fallback",
        "evaluation operation without guaranteed ONNX export"),
    "boxes_iou_bev": (
        "boxes_a[N,7], boxes_b[M,7] with x,y,z,dx,dy,dz,heading in radians",
        "same accelerator; native contiguous float32 box representation",
        "retain native empty shape/restrictions and zero-size-kernel behavior",
        "no complete native BEV backward contract",
        "no guaranteed symbolic"),
    "boxes_iou3d_gpu": (
        "boxes_a[N,7], boxes_b[M,7]; original BEV intersection and height/angle conventions",
        "same accelerator; native contiguous float32 boxes",
        "retain native empty shape/restrictions",
        "no complete differentiable BEV kernel; do not advertise full box gradients",
        "no guaranteed symbolic"),
    "nms_gpu": (
        "boxes[N,7], scores[N], thresh, optional pre_maxsize; descending score sort and rotated BEV suppression",
        "same-device int64 kept indices; original native float32 boxes",
        "preserve original empty/preselection behavior; no claimed support beyond supplier restrictions",
        "no mutation or gradient contract; return (indices,None), not just indices",
        "no guaranteed symbolic"),
    "nms_normal_gpu": (
        "boxes[N,7], scores[N], thresh; original non-rotated suppression, not interchangeable with nms_gpu",
        "same-device int64 kept indices; native contiguous box/score layout",
        "preserve original empty behavior and exceptions",
        "no mutation or gradient contract; return (indices,None)",
        "no guaranteed symbolic"),
    "points_in_boxes_cpu": (
        "points[N,3], boxes[M,7]=center,size,heading; preserve original inclusion boundaries",
        "explicit CPU float32 inputs; int32 membership[M,N]",
        "retain zero-dimensional membership axes and supplier restrictions",
        "no mutation or gradients",
        "CPU evaluation API, no guaranteed symbolic"),
    "points_in_boxes_gpu": (
        "points[B,N,3], boxes[B,M,7]; same batch; first matching box index or -1",
        "same accelerator; contiguous float32; output int32[B,N]",
        "retain native empty-batch/point/box restrictions and -1 initialization",
        "no gradients or input mutation",
        "no guaranteed symbolic"),
    "roiaware_pool3d": (
        "rois[R,7], pts[P,3], pts_feature[P,C], out_size int/tuple, max_pts_each_voxel, max/avg mode",
        "same accelerator; contiguous float32; output[R,X,Y,Z,C]",
        "retain empty-voxel initialization and native empty-input restrictions",
        "saves voxel point indices/argmax for pts_feature backward only",
        "no guaranteed symbolic"),
    "roipoint_pool3d": (
        "points[B,N,3], point_features[B,N,C], boxes3d[B,M,7], pool_extra_width, num_sampled_points S",
        "same accelerator; output[B,M,S,3+C] plus int32 empty flags[B,M]",
        "retain empty-box flags and original sampling/padding rules; test zero dimensions separately",
        "forward-only upstream contract; backward raises, never invent gradients",
        "no guaranteed symbolic"),
    "rotate_iou_eval": (
        "NumPy boxes[N,5], query_boxes[K,5]=x,y,dx,dy,angle; criterion=-1/0/1/other; device_id",
        "NumPy inputs/output; complete Numba CUDA evaluation operation, not a public CUDA kernel handle",
        "N=0 or K=0 returns NumPy zeros[N,K]",
        "no mutation or gradients; vendor implements whole operation",
        "evaluation-only; no ONNX symbolic"),
    "torchvision_nms": (
        "boxes[N,4], scores[N], iou_threshold; three positional arguments; exclusive coordinates",
        "same device/dtype; scores sorted internally; output int64[K]",
        "N=0 returns same-device int64[0]",
        "no mutation or backward contract; ties need not be bitwise identical",
        "preserve existing exporter behavior; no new engine-specific rewrite"),
}

RUNTIME_SEMANTICS = {
    "__init__": "Accept device=None/string/torch.device; reject foreign namespaces; no allocation or health check on ordinary import.",
    "api": "Return the selected torch device namespace; CPU returns None; missing SDK raises.",
    "device": "Return torch.device with current logical index and visible-device mapping, not a physical PCI index.",
    "is_available": "Return actual runtime availability, not merely SDK importability.",
    "device_count": "Return visible logical device count; CPU test backend returns 1.",
    "current_device": "Return current logical index; CPU returns 0.",
    "resolve_device": "Resolve None/string/device/int without changing binding; reject foreign device types.",
    "set_device": "Bind this process to logical device/index; CPU no-op.",
    "is_accelerator_tensor": "Return bool from tensor.device.type; CPU tensors are not accelerator tensors.",
    "validate_environment": "Check python/package specifiers; report installed versions; never install or replace vendor Torch.",
    "to_device": "Move Tensor/Module and recursive mapping/list/tuple/namedtuple; retain dtype, graph and parameter identity as PyTorch .to allows; honor non_blocking.",
    "float_tensor": "Legacy size/data constructors; torch.Size is a shape; current-device float32, preserve kwargs.",
    "int_tensor": "Legacy size/data constructors; current-device int32, preserve kwargs.",
    "manual_seed": "Seed the original device scope; do not silently expand to all devices; optional numerical RNG reseeds too.",
    "get_rng_state": "Return vendor RNG state or explicit experimental state envelope; not cross-vendor bitwise equivalence.",
    "set_rng_state": "Restore the matching state and optional numerical counter; invalid/foreign state must not be ignored.",
    "synchronize": "Wait for selected device/stream per vendor API; CPU no-op; preserve exceptions.",
    "empty_cache": "Release unused allocator cache only, not live tensors; CPU no-op.",
    "reset_peak_memory_stats": "Reset allocator peak statistics for selected runtime; CPU no-op.",
    "memory_allocated": "Allocated bytes on requested/current device; CPU returns 0.",
    "max_memory_allocated": "Peak allocated bytes; CPU returns 0.",
    "get_device_properties": "Return vendor properties including total_memory where required; CPU raises explicit capability error.",
    "autocast": "Context manager with enabled/dtype; reject unavailable device or unsupported dtype; training and inference choose separate overrides.",
    "grad_scaler": "Training-only compatible scale/step/update/state_dict/load_state_dict; validation must not step/update it.",
    "configure_convolution": "Set vendor convolution-library policy; compatible CUDA-shaped controls are explicitly shared.",
    "convolution_flags": "Temporary convolution policy context restoring prior state even on exception.",
    "health_check": "Allocate, compute and synchronize on target hardware; return True or raise, never CPU fallback.",
    "call": "Dispatch a complete semantic operation through the initialized purpose's frozen registry; preserve outputs/autograd.",
    "dependency": "Lazily load the named sparse/link/quantization/pruning provider or fail with backend and reason.",
    "capabilities": "Report registration, declared dependency checks, successful execution count and hardware validation separately.",
    "numerics_state": "Report native/portable_v1, protocol, seed/counter and executed numerical operation counts; not an accuracy pass.",
}


def _signature(path, attribute):
    nodes = ast.parse(path.read_text(encoding="utf-8")).body
    for name in attribute.split("."):
        name = "forward" if name == "apply" else name
        node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
        nodes = node.body
    args = copy.deepcopy(node.args)
    if args.args and args.args[0].arg in ("self", "ctx"):
        args.args.pop(0)
    return ast.unparse(args)


def build_contract():
    from . import PlatformContext, configure_platform, get_platform, register_backend, register_op, capabilities, require
    from .common.runtime import DeviceBackend
    from .training.backend import TrainingBackend
    from .inference.backend import InferenceBackend

    contract = json.loads((ROOT / "common/legacy_contract.json").read_text(encoding="utf-8"))
    baseline = {item["name"] for item in contract["operations"]}
    if baseline != {name for _, name in OPERATIONS}:
        raise ValueError("Registered operations differ from documented semantic contracts")
    runtime_members = {name for name, value in vars(DeviceBackend).items()
                       if (not name.startswith("_") or name == "__init__")
                       and (inspect.isfunction(value) or isinstance(value, property))}
    expected_runtime = {item["name"] for item in contract["runtime"]} - {"autocast", "grad_scaler", "call", "capabilities"}
    if runtime_members != expected_runtime:
        raise ValueError("DeviceBackend public API differs from the documented migration contract")
    for item in contract["operations"]:
        name = item["name"]
        item["legacy_reference_source"] = item["reference_source"]
        item["scope"] = "training" if name in TRAINING_ONLY else "common"
        item["purposes"] = ["training"] if name in TRAINING_ONLY else ["training", "inference"]
        item["legacy_signature"] = item["signature"]
        if name in NATIVE_OPERATIONS:
            module, attribute, _ = NATIVE_OPERATIONS[name]
            path = ROOT / "backends/nvidia/ops" / (module + ".py")
            item["signature"] = name + "(" + _signature(path, attribute) + ")"
            item["reference_source"] = path.relative_to(ROOT).as_posix()
        elif name.startswith("sparse."):
            item["reference_source"] = "shared/sparse.py"
        elif name.startswith("distributed."):
            item["reference_source"] = "training/distributed.py" if name == "distributed.wrap_model" else "common/collectives.py"
        elif "batch_norm" in name:
            item["reference_source"] = "shared/layers.py"
        elif name == "torchvision_nms":
            item["reference_source"] = "shared/portable.py; backends/nvidia/ops/vision.py"
        else:
            item["reference_source"] = "backends/nvidia/legacy.py"
        if name in NATIVE_SEMANTICS:
            inputs, layout, empty, state, export = NATIVE_SEMANTICS[name]
        elif item["category"] == "distributed":
            inputs = item["signature"] + "; standard supported PyTorch process-group, rank, reduction and async Work semantics"
            layout = "collectives use the process group's declared device; CPU Gloo/object groups remain explicit; DDP belongs to training"
            empty = "preserve pinned PyTorch/transport zero-count collective restrictions; never skip participation on one rank"
            state = "preserve process-group lifecycle, in-place tensor/list writes and Work completion; wrap_model retains parameter identities"
            export = "host orchestration, not an ONNX graph operator"
        elif item["category"] == "sparse":
            inputs = item["signature"] + "; features[A,C], supplier indices/spatial_shape/batch_size; preserve geometry and ordering"
            layout = "same device; supplier sparse representation and parameter layout; no new wrapper parameter prefix"
            empty = "preserve supplier zero-active-site support/restrictions; test constructor, replacement, dense conversion and backward"
            state = "replace_feature preserves geometry; features/to_dense retain graph; layer factories preserve state_dict keys"
            export = "no generic sparse exporter promised; supplier must explicitly support required export path"
        elif "batch_norm" in name:
            inputs = item["signature"] + "; input[N,C,...], affine[C], running statistics[C], eps/momentum/group as applicable"
            layout = "input device/layout/dtype retained; group and local sample-count semantics follow supplier"
            empty = "preserve global/local empty-batch synchronization rules; never fake local gradients"
            state = "training updates running mean/variance/count; eval reads statistics without updating them; complete declared affine/input backward"
            export = "preserve eval normalization behavior; distributed training collectives are not export nodes"
        elif name.startswith("legacy_link."):
            inputs = item["signature"] + "; exact pinned spring.linklink signature; no universal SDK ABI is claimed"
            layout = "exact supplier tensor/communicator/device layout; optional dependency loaded only on use"
            empty = "pinned SDK contract; do not claim zero-count support without execution tests"
            state = "preserve supplier initialization, groups, in-place writes and async handles; no invented gradient semantics"
            export = "SDK host orchestration, not an ONNX graph operation"
        else:
            inputs = item["signature"] + "; exact optional supplier optimizer/precision/quantization configuration"
            layout = "supplier parameter objects, groups/master tensors and backend enum representation remain unchanged"
            empty = "retain pinned supplier empty-module/parameter-group validation and exceptions"
            state = "optimizer construction/reload or explicit precision-module registration only; UP owns hyperparameters and training steps"
            export = "host configuration/optimizer state, not a numeric ONNX graph operation"
        item.update(inputs=inputs, device_and_layout=layout, empty_input=empty, state_updates=state, export=export)
        item["registered_providers"] = {}
        for (owner, op_name), record in sorted(OPERATIONS.items()):
            if op_name != name:
                continue
            scopes = record.get("implementations") or {record.get("scope", "common"): record}
            item["registered_providers"][owner] = {
                scope: {key: entry.get(key) for key in ("backward", "dtypes", "variants", "dependencies", "conventions")}
                for scope, entry in scopes.items()}
    for item in contract["runtime"]:
        name = item["name"]
        owner = (TrainingBackend if name == "grad_scaler" else PlatformContext if name in ("call", "capabilities")
                 else TrainingBackend if name == "autocast" else DeviceBackend)
        member = getattr(owner, name)
        item["signature"] = name if isinstance(member, property) else name + str(inspect.signature(member))
        item["owner"] = "execution.autocast (training or inference)" if name == "autocast" else owner.__name__
        item["contract"] = RUNTIME_SEMANTICS[name]
    contract["entry_points"] = [{"name": value.__name__, "signature": str(inspect.signature(value))}
                                for value in (configure_platform, get_platform, register_backend, register_op, capabilities, require)]
    contract["execution_interfaces"] = {
        "training": {name: str(inspect.signature(getattr(TrainingBackend, name))) for name in
                     ("autocast", "grad_scaler", "wrap_model", "wrap_legacy_model", "reduce_gradients",
                      "build_fused_optimizer", "reload_fused_optimizer", "call")},
        "inference": {name: str(inspect.signature(getattr(InferenceBackend, name))) for name in
                      ("autocast", "export_onnx", "call")}}
    contract.update(contract_snapshot_version=2, runtime_class="DeviceBackend + purpose execution + PlatformContext",
                    status="structural migration contract, not a new hardware validation result")
    return contract
