"""Rebuild the source-auditable hardware inventory from the frozen Git revision."""
import argparse
import io
import json
import subprocess
import tarfile
from pathlib import Path

from check_hardware_boundaries import ROOT, check_source, check_tree

UPSTREAM = "52aaeb9ed27a490820696d7754d19f98cd6ecd11"
ROUTES = {
    "hardware_runtime": "get_accelerator runtime/AMP/resource methods",
    "hardware_runtime_import": "get_accelerator runtime methods",
    "device_transfer": "device-neutral .to(get_accelerator().device)",
    "device_predicate": "get_accelerator().is_accelerator_tensor",
    "hardcoded_device": "selected backend device",
    "hardware_literal": "backend configuration or collective device policy",
    "collective_bypass": "up.accelerator.distributed",
    "distributed_wrapper_bypass": "up.accelerator.distributed.wrap_model",
    "vendor_import": "accelerator layers, sparse interface or backend provider",
    "native_extension_import": "ops semantic dispatch and deferred NVIDIA loader",
    "cuda_jit_import": "ops.rotate_iou_eval and NVIDIA evaluator implementation",
    "dynamic_vendor_import": "backend dependency provider",
    "sparse_tensor_access": "accelerator.sparse tensor operations",
    "amp_bypass": "get_accelerator.autocast/grad_scaler",
    "layer_bypass": "accelerator.layers factories",
    "specialized_op_import": "ops semantic dispatch for compiled vision operations",
    "specialized_op_bypass": "ops semantic dispatch for compiled vision operations",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=UPSTREAM)
    parser.add_argument("--output", default=str(ROOT / "docs/hardware/upstream_inventory.json"))
    args = parser.parse_args()
    archive = subprocess.check_output(["git", "archive", args.ref], cwd=ROOT)
    sites, native = [], {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for member in bundle.getmembers():
            if not member.isfile():
                continue
            path = member.name
            if path.startswith("up/extensions/csrc/"):
                family = path.split("/")[3]
                native.setdefault(family, []).append(path)
            if not path.startswith("up/") or not path.endswith(".py"):
                continue
            text = bundle.extractfile(member).read().decode("utf8")
            lines = text.splitlines()
            for issue in check_source(text, path):
                location = path
                if path.startswith("up/extensions/python/"):
                    location = "up/accelerator/backends/nvidia_ops/" + Path(path).name
                if path.endswith("kitti_object_eval_python/rotate_iou.py"):
                    location = "up/accelerator/backends/nvidia_ops/rotate_iou.py"
                sites.append(dict(issue, source=lines[issue["line"] - 1].strip(),
                                  route=ROUTES[issue["rule"]],
                                  implementation_or_facade=location, status="isolated"))
    violations = check_tree()
    report = {
        "upstream_commit": args.ref, "python_hardware_sites": len(sites),
        "current_boundary_violations": violations, "sites": sites,
        "native_operation_families": native,
        "native_loading_policy": "only the selected backend may load its own binary",
        "compatibility_entries": {
            "ModelHelper.cuda": "legacy name forwards to selected backend",
            "up.extensions": "legacy modules and Function.apply forward to ops",
            "matcher mode='cuda'": "legacy strategy label; no hardware binding",
            "CPU export stubs": "export-only, never registered as CPU numerical implementations",
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print("Hardware sites:", len(sites), "Native families:", len(native), "Violations:", len(violations))
    raise SystemExit(bool(violations))


if __name__ == "__main__":
    main()
