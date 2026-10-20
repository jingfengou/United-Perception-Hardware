"""Collect bounded validation evidence and source hashes without modifying tests."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path, normalize_newlines=False):
    data = path.read_bytes()
    if normalize_newlines:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def source_manifest(root):
    paths = [p for directory in ("up", "tests", "tools") for p in (root / directory).rglob("*.py")]
    paths += [root / name for name in (
        "setup.py", "MANIFEST.in", "pytest.ini", ".github/workflows/hardware-boundaries.yml")]
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(paths)}


def main(args):
    if args.verify_source:
        report = json.loads(Path(args.verify_source).read_text())
        actual = source_manifest(ROOT)
        expected = report["source_manifest"]
        differences = sorted(name for name in set(actual) | set(expected)
                             if actual.get(name) != expected.get(name))
        print(json.dumps(dict(source_match=not differences, files=len(actual), differences=differences), indent=2))
        raise SystemExit(bool(differences))

    import torch
    from check_hardware_boundaries import check_tree
    results, baseline = Path(args.results), Path(args.baseline)
    names = ("native_comparison.final.json", "models.final.json",
             "distributed.final.json", "onnx_comparison.json")
    reports = {name: json.loads((results / name).read_text()) for name in names}
    suites = ET.parse(results / "unit.final.xml").getroot().findall(".//testsuite")
    unit = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
            for key in ("tests", "errors", "failures", "skipped")}
    native_sources = {
        p.relative_to(ROOT).as_posix(): digest(p, normalize_newlines=True)
        for p in sorted((ROOT / "up/extensions/csrc").rglob("*")) if p.is_file()}
    native_differences = [
        name for name, sha in native_sources.items()
        if not (baseline / name).is_file() or digest(baseline / name, normalize_newlines=True) != sha]
    package_versions = {}
    for package in ("torch", "torchvision", "numpy", "numba", "numba-cuda", "onnx",
                    "torch-npu", "torch-mlu", "spconv", "mqbench", "msbench", "linklink"):
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            package_versions[package] = None
    artifact_names = list(names) + [
        "unit.final.xml", "unit.final.log", "routed_ops.final.log", "upstream_ops.log",
        "upstream_ops.pt", "routed_ops.final.pt", "models.final.log", "distributed.final.log",
        "onnx_upstream.json", "onnx_routed.json", "onnx_upstream.log", "onnx_routed.log",
        "baseline_build.log", "changed_build.final.log",
    ]
    artifacts = {name: digest(results / name) for name in artifact_names}
    boundaries = check_tree(ROOT)
    source = source_manifest(ROOT)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": "52aaeb9ed27a490820696d7754d19f98cd6ecd11",
        "execution_host_alias": "A800-1",
        "scope": "bounded synthetic regression, not dataset accuracy/convergence or full Runner acceptance",
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "packages": package_versions,
            "gpus": subprocess.check_output([
                "nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader"], text=True).strip().splitlines(),
        },
        "checks": {
            "unit": unit, "boundary_violations": boundaries,
            "native_comparison": reports["native_comparison.final.json"],
            "models": {name: {key: entry.get(key) for key in (
                "finite_gradients", "inference", "checkpoint_restore",
                "legacy_helper_state_and_forward", "amp_and_scaler")}
                for name, entry in reports["models.final.json"]["models"].items()},
            "distributed": reports["distributed.final.json"],
            "onnx_comparison": reports["onnx_comparison.json"],
            "native_source_differences": native_differences,
        },
        "automated_checks_passed": (
            not unit["errors"] and not unit["failures"] and not boundaries and not native_differences
            and all(report["passed"] for report in reports.values())),
        "hardware_status": {
            "nvidia_a800": "bounded_real_hardware_tests_passed",
            "ascend_910b": "runtime_entry_only_not_hardware_validated",
            "cambricon_mlu590": "runtime_entry_only_not_hardware_validated",
            "metax_c500": "runtime_entry_only_not_hardware_validated",
        },
        "limitations": [
            "Model comparison uses upstream ModelHelper with current registered components, not the entire original stack",
            "No full dataset/Runner accuracy, convergence, or performance validation",
            "No real optional linklink, spconv, MQBench, or MSBench execution",
            "Upstream PSRoIAlign CPU export kernel is absent; accelerator export symbols are compared",
            "CI Python/PyTorch matrix is configured but has not been executed in this session",
        ],
        "source_manifest": source,
        "source_manifest_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        "native_source_sha256": native_sources,
        "native_source_hash_normalization": "CRLF normalized to LF for source comparison",
        "native_binary_sha256": {
            name: {p.name: digest(p) for p in (root / "up/extensions").glob("_C*.so")}
            for name, root in (("baseline", baseline), ("changed", ROOT))},
        "artifact_sha256": artifacts,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(automated_checks_passed=report["automated_checks_passed"],
                         unit=unit, native_cases=reports["native_comparison.final.json"]["compared_cases"],
                         models=list(report["checks"]["models"]),
                         source_files=len(source), native_source_differences=native_differences), indent=2))
    raise SystemExit(not report["automated_checks_passed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results")
    parser.add_argument("--baseline")
    parser.add_argument("--output")
    parser.add_argument("--verify-source")
    main(parser.parse_args())
