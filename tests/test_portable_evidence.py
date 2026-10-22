"""CPU-only integrity tests for diagnostic traces and frozen source snapshots."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmark/hardware" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor(raw="raw", canonical="canonical", device="cuda"):
    return {"shape": [2], "dtype": "torch.float32", "device_type": device,
            "sha256": raw, "zero_canonical_sha256": canonical}


def row(output, operation="aten.mul.Tensor"):
    return {"index": 0, "phase": "backward", "operation": operation,
            "inputs": [[tensor()], {}], "output": output}


def test_signed_zero_hash_difference_is_not_nonzero_value_difference():
    result = tool("compare_dispatch_traces").compare([row(tensor("positive-zero"))],
                                                     [row(tensor("negative-zero"))])
    assert result["different_tensors"] == {"backward/torch.float32": 1}
    assert result["first_different_nonzero_fp32_values"] == {}


def test_true_fp32_difference_records_inputs_and_operation():
    result = tool("compare_dispatch_traces").compare([row(tensor("a", "a"))], [row(tensor("b", "b"))])
    first = result["first_different_nonzero_fp32_values"]["backward"]
    assert first["operation"] == "aten.mul.Tensor" and first["inputs_equal_ignoring_zero_signs"]


@pytest.mark.parametrize("right", [[], [row(tensor(), "aten.add.Tensor")], [row([tensor()])]])
def test_trace_comparison_rejects_unaligned_counts_operations_or_structure(right):
    with pytest.raises(ValueError):
        tool("compare_dispatch_traces").compare([row(tensor())], right)


def test_numeric_filter_reports_every_omission():
    rows = [row(tensor(device="cpu")), row(tensor(), "aten.detach.default"), row(tensor())]
    selected, omitted = tool("compare_dispatch_traces").numerical_rows(rows)
    assert selected == rows[-1:] and len(omitted) == 2


def test_source_snapshot_normalizes_only_newlines_and_rejects_overwrite(tmp_path):
    source = tmp_path / "source"
    (source / "up").mkdir(parents=True)
    content = b"VALUE = 1\r\n"
    (source / "up" / "example.py").write_bytes(content)
    for relative in ("packages/up-platform/src/up_platform/common/runtime.py",
                     "packages/up-platform/src/up_platform/backends/nvidia/csrc/example.cu",
                     "benchmark/hardware/example.py"):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    archive = tmp_path / "nvidia-source.tar.gz"
    command = [sys.executable, str(ROOT / "tools/platform/snapshot_hardware_source.py"),
               "--source", str(source), "--output", str(archive)]
    subprocess.run(command, check=True, capture_output=True)
    manifest, _, index, _ = tool("summarize_portable_cohort").read_source(tmp_path, "nvidia")
    assert index["files"]["up/example.py"] == hashlib.sha256(content).hexdigest()
    assert index["normalized_up_files"]["up/example.py"] == hashlib.sha256(b"VALUE = 1\n").hexdigest()
    assert index["source_layout"] == "up-platform-v1"
    assert len(index["files"]) == 4
    assert manifest == index["normalized_up_manifest_sha256"]
    before = archive.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert archive.read_bytes() == before
    index["source_layout"] = "untrusted-layout"
    archive.with_name(archive.name + ".index.json").write_text(json.dumps(index))
    with pytest.raises(AssertionError, match="Unknown source snapshot layout"):
        tool("summarize_portable_cohort").read_source(tmp_path, "nvidia")


@pytest.mark.parametrize("name,kind", [("../up/escape.py", "file"), ("/up/escape.py", "file"),
                                       ("up/symlink.py", "symlink")])
def test_source_reader_rejects_unsafe_archive_entries(tmp_path, name, kind):
    archive = tmp_path / "nvidia-source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo(name)
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "../private"
            stream.addfile(member)
        else:
            member.size = 1
            stream.addfile(member, io.BytesIO(b"x"))
    archive.with_name(archive.name + ".index.json").write_text(json.dumps({
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}))
    with pytest.raises(AssertionError):
        tool("summarize_portable_cohort").read_source(tmp_path, "nvidia")


def test_statistics_refuses_to_replace_existing_result(tmp_path):
    existing = tmp_path / "summary.json"
    existing.write_text("original\n")
    with pytest.raises(ValueError, match="Preserve"):
        tool("summarize_portable_cohort").write_statistics(tmp_path, None)
    assert existing.read_text() == "original\n"


def test_diagnostic_package_has_explicit_json_only_allowlist():
    reports = tool("package_portable_diagnostics").REPORTS
    assert len(set(reports)) == len(reports)
    assert all(Path(name).name == name and name.endswith(".json") for name in reports)
