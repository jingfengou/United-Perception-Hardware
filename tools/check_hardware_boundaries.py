"""Check executable Python AST nodes, excluding hardware implementations."""
import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR_MODULES = (
    "spring.linklink", "spconv", "mqbench", "msbench", "torch_npu", "torch_mlu",
    "cupy", "tensorrt", "pynvml",
)
RUNTIME_PREFIXES = ("torch.cuda", "torch.npu", "torch.mlu", "torch.backends")
IMPLEMENTATIONS = ("up.accelerator.backends", "up.accelerator.native",
                   "up.extensions._C", "up_extensions", "up.extensions.ext")
COMPATIBILITY_IMPORTS = {"up/extensions/ext.py"}
VISION_NATIVE_OPS = (
    "nms", "batched_nms", "roi_align", "roi_pool", "deform_conv2d", "ps_roi_align", "ps_roi_pool",
    "RoIAlign", "RoIPool", "DeformConv2d", "PSRoIAlign", "PSRoIPool",
)
# These are strategy labels retained for callers, not device selection.
LEGACY_DEFAULTS = {
    ("up/tasks/det/models/utils/matcher.py", "mode", "cuda"),
    ("up/tasks/det/plugins/yolox/models/postprocess/roi_supervisor.py", "mode", "cuda"),
}


def check_source(text, path):
    tree = ast.parse(text)
    aliases = {}
    issues = []
    allowed_defaults = set()

    def report(node, rule):
        issues.append({"path": path, "line": node.lineno, "rule": rule})

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args.args
            for argument, default in zip(arguments[len(arguments) - len(node.args.defaults):], node.args.defaults):
                if isinstance(default, ast.Constant) and (path, argument.arg, default.value) in LEGACY_DEFAULTS:
                    allowed_defaults.add(id(default))
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name
                if any(alias.name == vendor or alias.name.startswith(vendor + ".") for vendor in VENDOR_MODULES):
                    report(node, "vendor_import")
                if alias.name.startswith(RUNTIME_PREFIXES):
                    report(node, "hardware_runtime_import")
                if path not in COMPATIBILITY_IMPORTS and alias.name.startswith(IMPLEMENTATIONS):
                    report(node, "native_extension_import")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                full = module + "." + alias.name
                aliases[alias.asname or alias.name] = full
                if any(full == vendor or full.startswith(vendor + ".") for vendor in VENDOR_MODULES):
                    report(node, "vendor_import")
                if full.startswith(RUNTIME_PREFIXES):
                    report(node, "hardware_runtime_import")
                if path not in COMPATIBILITY_IMPORTS and full.startswith(IMPLEMENTATIONS):
                    report(node, "native_extension_import")
                if alias.name == "_C" and ("extensions" in module or node.level):
                    report(node, "native_extension_import")
                if module in ("ext", "up.extensions.ext") and path.startswith("up/extensions/python/"):
                    report(node, "native_extension_import")
                if module == "numba" and alias.name == "cuda":
                    report(node, "cuda_jit_import")
                if module.startswith("torchvision.ops") and alias.name in VISION_NATIVE_OPS:
                    report(node, "specialized_op_import")

    def qualified(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return qualified(node.value) + "." + node.attr
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in allowed_defaults:
            if node.value.split(":")[0] in ("cuda", "npu", "mlu", "nccl", "hccl", "cncl"):
                report(node, "hardware_literal")
        if isinstance(node, ast.Attribute):
            full = qualified(node)
            if full.startswith(RUNTIME_PREFIXES):
                report(node, "hardware_runtime")
            if node.attr in ("is_cuda", "is_npu", "is_mlu"):
                report(node, "device_predicate")
            if (path in ("up/tasks/det_3d/models/backbones_3d/spconv_backbone.py",
                         "up/tasks/det_3d/models/backbones_3d/map_to_bev.py")
                    and node.attr in ("features", "replace_feature", "dense")
                    and qualified(node.value) not in ("spconv", "up.accelerator.sparse")):
                report(node, "sparse_tensor_access")
        if isinstance(node, ast.Call):
            full = qualified(node.func)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda":
                report(node, "device_transfer")
            if full.startswith("torch.distributed."):
                report(node, "collective_bypass")
            if full == "torch.nn.parallel.DistributedDataParallel":
                report(node, "distributed_wrapper_bypass")
            if full in ("torch.autocast", "torch.amp.autocast", "torch.amp.GradScaler"):
                report(node, "amp_bypass")
            if full == "torch.nn.SyncBatchNorm":
                report(node, "layer_bypass")
            if full.startswith("torchvision.ops.") and full.rsplit(".", 1)[-1] in VISION_NATIVE_OPS:
                report(node, "specialized_op_bypass")
            if full == "importlib.import_module" and node.args and isinstance(node.args[0], ast.Constant):
                name = str(node.args[0].value)
                if any(name == vendor or name.startswith(vendor + ".") for vendor in VENDOR_MODULES):
                    report(node, "dynamic_vendor_import")
            for keyword in node.keywords:
                if keyword.arg == "device" and isinstance(keyword.value, ast.Constant):
                    value = keyword.value.value
                    if isinstance(value, str) and value.split(":")[0] in ("cuda", "npu", "mlu"):
                        report(node, "hardcoded_device")
            if full == "torch.device" and node.args and isinstance(node.args[0], ast.Constant):
                if str(node.args[0].value).split(":")[0] in ("cuda", "npu", "mlu"):
                    report(node, "hardcoded_device")
    return sorted({(item["path"], item["line"], item["rule"]): item for item in issues}.values(),
                  key=lambda item: (item["path"], item["line"], item["rule"]))


def check_tree(root=ROOT):
    issues = []
    for path in sorted((root / "up").rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if relative.startswith("up/accelerator/"):
            continue
        issues.extend(check_source(path.read_text(encoding="utf8"), relative))
    return issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    findings = check_tree()
    if args.json:
        print(json.dumps(findings, indent=2))
    else:
        for issue in findings:
            print("{path}:{line}: {rule}".format(**issue))
        print("Hardware boundary violations:", len(findings))
    raise SystemExit(bool(findings))
