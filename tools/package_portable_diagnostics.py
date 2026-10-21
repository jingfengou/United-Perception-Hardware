"""Preserve and check portable-profile diagnostics, not full-training acceptance."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET


REPORTS = (
    "nvidia-convolution-routing-v4.json", "nvidia-convolution-routing-v5.json",
    "metax-convolution-routing-v5.json", "nvidia-dropout-backward.json",
    "metax-dropout-backward.json", "nvidia-primitives-v10.json", "metax-primitives-v10.json",
    "nvidia-rng-state-control.json", "metax-rng-state-control.json",
    "nvidia-early-v8-yolo4.json", "metax-early-v8-yolo4.json",
    "nvidia-early-v10-detr64.json", "metax-early-v10-detr64.json",
    "nvidia-early-v10-vit32.json", "metax-early-v10-vit32.json",
    "nvidia-v8-detr9-trace.json", "metax-v8-detr9-trace.json",
    "detr9-v8-trace-comparison.json",
    "nvidia-v8-vit1-trace.json", "metax-v8-vit1-trace.json",
    "vit1-v8-trace-comparison.json", "nvidia-yolo1-trace.json", "metax-yolo1-trace.json",
    "yolo1-trace-comparison.json", "yolo1-current-trace-comparison.json",
)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def check(root, read):
    snapshots = sibling("summarize_portable_cohort")
    traces = sibling("compare_dispatch_traces")
    sources = {}
    snapshot_rows = {}
    for revision, directory, suffix in (("v8", root / "diagnostics", "-v8"),
                                        ("v10", root / "diagnostics", "-v10"),
                                        ("v11", root / "fixed-order", "")):
        pair = [snapshots.read_source(directory, backend + suffix) for backend in ("nvidia", "metax")]
        assert pair[0][0] == pair[1][0], "Different diagnostic application sources"
        sources[revision] = pair[0][0]
        snapshot_rows[revision] = pair[0][2]["normalized_up_files"]
    for version in ("v8", "v10"):
        probe = load(read("nvidia-early-" + version + ("-yolo4.json" if version == "v8" else "-detr64.json")))
        index = load(root / "diagnostics" / ("nvidia-" + version + "-source.tar.gz.index.json"))
        assert probe["tool_sha256"] == index["files"]["tools/probe_cross_vendor_numerics.py"]
    changed_v10_v11 = sorted(name for name in snapshot_rows["v11"]
                             if snapshot_rows["v10"].get(name) != snapshot_rows["v11"][name])
    assert changed_v10_v11 == ["up/accelerator/numerics.py", "up/accelerator/runtime.py"]
    # The final revision adds capability-error context; kernel files must be unchanged.
    assert all(snapshot_rows["v10"][name] == value for name, value in snapshot_rows["v11"].items()
               if name.startswith("up/accelerator/backends/portable"))

    short = []
    for revision, name, steps in (("v8", "yolo4", 4), ("v10", "detr64", 64), ("v10", "vit32", 32)):
        a, b = [load(read(backend + "-early-" + revision + "-" + name + ".json"))
                for backend in ("nvidia", "metax")]
        assert a["status"] == b["status"] == "completed"
        assert a["initial_state"] == b["initial_state"] and a["optimizer"] == b["optimizer"]
        assert a["tool_sha256"] == b["tool_sha256"]
        assert len(a["steps"]) == len(b["steps"]) == steps
        for left, right in zip(a["steps"], b["steps"], strict=True):
            for key in ("step", "batch_sha256", "loss", "state_after_update", "rng_before_forward"):
                assert left[key] == right[key], "Short control mismatch: " + name + "/" + key
        short.append({"model": a["model"], "steps": steps, "source_revision": revision,
                      "losses_equal": True, "full_parameter_states_equal": True,
                      "inputs_and_pre_forward_rng_equal": True,
                      "report_sha256": [sha(read(backend + "-early-" + revision + "-" + name + ".json").read_bytes())
                                         for backend in ("nvidia", "metax")]})

    a, b = [load(read(backend + "-primitives-v10.json")) for backend in ("nvidia", "metax")]
    assert a["status"] == b["status"] == "passed" and a["tensor_hashes"] == b["tensor_hashes"]
    assert all(case.get("same_card_repeatable") is True
               for report in (a, b) for case in report["cases"].values() if isinstance(case, dict))
    primitive_tensors = len(a["tensor_hashes"])
    a, b = [load(read(backend + "-rng-state-control.json")) for backend in ("nvidia", "metax")]
    assert a["status"] == b["status"] == "passed" and a["cases"] == b["cases"]
    rng_cases = len(a["cases"])

    a, b = [load(root / "fixed-order" / (backend + "-math-contracts.json")) for backend in ("nvidia", "metax")]
    assert a["status"] == b["status"] == "passed" and a["cases"].keys() == b["cases"].keys()
    for name in a["cases"]:
        for key in ("inputs", "outputs"):
            assert a["cases"][name][key] == b["cases"][name][key]
    math_cases = len(a["cases"])
    cpu = {}
    for backend in ("nvidia", "metax"):
        suite = ET.parse(root / "fixed-order" / (backend + "-contracts-final.xml")).getroot().find("testsuite")
        cpu[backend] = {key: int(suite.attrib[key]) for key in ("tests", "failures", "errors", "skipped")}
        assert cpu[backend] == {"tests": 202, "failures": 0, "errors": 0, "skipped": 0}

    divergences = {}
    for name, prefix, comparison in (("detr", "v8-detr9", "detr9-v8"),
                                      ("vit", "v8-vit1", "vit1-v8"),
                                      ("yolo", "yolo1", "yolo1-current")):
        paths = [read(backend + "-" + prefix + "-trace.json") for backend in ("nvidia", "metax")]
        report = load(read(comparison + "-trace-comparison.json"))
        assert report["trace_sha256"] == [sha(path.read_bytes()) for path in paths]
        rows = [load(path) for path in paths]
        if report["numeric_only"]:
            rows = [traces.numerical_rows(value)[0] for value in rows]
        actual = traces.compare(*rows)
        for key, value in actual.items():
            assert report[key] == value, "Stale trace comparison: " + name
        divergences[name] = {"operations": report["operations"],
                             "numeric_only": report["numeric_only"],
                             "first_nonzero_fp32": {phase: {key: row[key] for key in
                                 ("index", "operation", "inputs_equal_ignoring_zero_signs")}
                                  for phase, row in report["first_different_nonzero_fp32_values"].items()},
                             "first_raw_difference": {phase: {key: row[key] for key in
                                 ("index", "operation", "inputs_equal")}
                                  for phase, row in report["first_different"].items()}}
    return {"status": "passed_diagnostic_checks_not_full_training_acceptance",
            "source_manifests": sources, "v10_v11_changed_source_files": changed_v10_v11,
            "v10_v11_portable_backend_kernels_equal": True, "short_controls": short,
            "gpu_math_cases_v11": math_cases, "primitive_tensor_hashes_equal_v10": primitive_tensors,
            "rng_protocol_cases_equal": rng_cases, "cpu_contracts_v11": cpu, "pre_fix_trace_divergences": divergences,
            "limitations": ["Short controls are not complete training", "No isolated performance benchmark",
                            "v4/v5 profiler source labels follow invocation provenance, not embedded source hashes",
                            "RNG unit restoration is not a complete interrupted Runner resume test"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--working", type=Path, required=True)
    args = parser.parse_args()
    destination = args.root / "diagnostics"
    archive_path = destination / "evidence.tar.gz"
    summary_path = destination / "summary.json"
    if archive_path.exists() or summary_path.exists():
        raise ValueError("Preserve published diagnostics; use a new destination")
    files = {}
    for name in REPORTS:
        path = args.working / name
        path.resolve().relative_to(args.working.resolve())
        if path.is_symlink() or not path.is_file():
            raise ValueError("Only regular explicitly selected JSON evidence is allowed")
        load(path)
        files[name] = sha(path.read_bytes())
    summary = check(args.root, lambda name: args.working / name)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as stream:
        for name in files:
            stream.add(args.working / name, arcname=name, recursive=False)
    index = {"archive_sha256": sha(archive_path.read_bytes()), "files": files,
             "collector_sha256": sha(Path(__file__).read_bytes()),
             "scope": "Historical diagnostic controls and full-tensor hashes, not full-training acceptance",
             "exclusions": ["weights", "data images", "SDK binaries", "private keys", "tensor payloads"]}
    archive_path.with_name(archive_path.name + ".index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"files": len(files), "gpu_math_cases": summary["gpu_math_cases_v11"],
                      "short_controls": len(summary["short_controls"]), "archive_sha256": index["archive_sha256"]}))


if __name__ == "__main__":
    main()
