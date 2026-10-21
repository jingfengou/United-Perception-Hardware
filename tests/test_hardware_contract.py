import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from up.accelerator.registry import OPERATIONS

ROOT = Path(__file__).resolve().parents[1]
TOOL = runpy.run_path(str(ROOT / "tools/export_accelerator_contract.py"))
build_contract, markdown = TOOL["build_contract"], TOOL["markdown"]


def test_complete_contract_matches_registered_operations():
    contract = build_contract(ROOT)
    assert {entry["name"] for entry in contract["operations"]} == {name for _, name in OPERATIONS}
    assert all(entry["signature"] and entry["returns"] and entry["gradients"]
               for entry in contract["operations"])


def test_contract_and_index_are_not_stale():
    contract = build_contract(ROOT)
    expected = json.loads((ROOT / "docs/hardware/interface_contract.json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps(contract)) == expected
    assert markdown(contract) == (ROOT / "docs/hardware/API_INDEX.md").read_text(encoding="utf-8")


def test_new_operation_requires_semantic_documentation(monkeypatch):
    monkeypatch.setitem(OPERATIONS, ("portable", "undocumented_test_operation"),
                        {"implementation": lambda: None, "backward": False, "dtypes": None, "variants": None})
    with pytest.raises(ValueError, match="Missing semantic documentation"):
        build_contract(ROOT)


def test_legacy_apply_signature_excludes_autograd_context():
    operations = {entry["name"]: entry for entry in build_contract(ROOT)["operations"]}
    assert operations["roi_align"]["signature"].startswith("roi_align(features, rois,")
    assert operations["roipoint_pool3d"]["gradients"] == "none; upstream backward raises"


def test_export_does_not_load_native_or_vendor_modules():
    script = """
import sys
import runpy
runpy.run_path('tools/export_accelerator_contract.py')['build_contract']()
assert not any(name.startswith(('torch_npu', 'torch_mlu', 'spconv', 'spring',
                               'mqbench', 'msbench', 'up.extensions._C', 'up.tasks'))
               for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)
