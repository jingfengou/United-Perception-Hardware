"""Source/build checks for the LightX2V-style structural migration."""
import json
from pathlib import Path
import re
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_preserved_math_and_protected_up_sources():
    check = runpy.run_path(str(ROOT / "tools/platform/check_project.py"))
    assert check["check_protected"]() == 665
    assert check["check_layout"]() == 80


def test_python_definition_guard_detects_math_changes_but_not_import_moves():
    check = runpy.run_path(str(ROOT / "tools/platform/check_project.py"))
    digest = check["definition_digest"]
    original = "from old import op\ndef forward(x):\n    return op(x) + 1\n"
    relocated = original.replace("from old", "from new")
    assert digest(original) == digest(relocated)
    assert digest(original) != digest(relocated.replace("+ 1", "+ 2"))


def test_nvidia_build_keeps_binary_abi_and_all_native_sources(monkeypatch, tmp_path):
    import torch
    import torch.utils.cpp_extension as extension
    monkeypatch.setenv("UP_BUILD_BACKEND", "nvidia")
    monkeypatch.setattr(torch.version, "cuda", "12.6")
    monkeypatch.setattr(extension, "CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(extension, "CUDAExtension", lambda name, sources, **kwargs: dict(name=name, sources=sources, **kwargs))
    setup = runpy.run_path(str(ROOT / "packages/up-platform/setup.py"))
    native, = setup["get_extensions"]()
    assert native["name"] == "up_platform.backends.nvidia._C"
    package = ROOT / "packages/up-platform"
    layout = json.loads((ROOT / "docs/architecture/platform-layout.json").read_text(encoding="utf-8"))
    expected = {str((ROOT / target).relative_to(package)) for old, target in layout["relocations"].items()
                if "/nvidia/csrc/" in old and target.endswith((".cu", ".cpp"))}
    assert {str(Path(name)) for name in native["sources"]} == expected
    includes = [Path(path) for path in native["include_dirs"]]
    for name in native["sources"]:
        source = package / name
        for header in re.findall(r'^#include "([^"]+)"', source.read_text(encoding="utf-8"), re.M):
            assert any((directory / header).is_file() for directory in [source.parent] + includes), (name, header)


def test_upper_layer_cannot_import_category_vendor_implementations():
    check = runpy.run_path(str(ROOT / "tools/platform/check_hardware_boundaries.py"))
    issues = check["check_source"]("from up_platform.ops.nms.nvidia.nms import naive_nms\n", "up/model.py")
    assert any(item["rule"] == "native_extension_import" for item in issues)
    assert not check["check_source"]("from up_platform import ops\n", "up/model.py")
