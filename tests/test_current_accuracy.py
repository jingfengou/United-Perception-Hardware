"""The one-seed validation cannot silently change budget, vendor or output stage."""
from argparse import Namespace
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tool(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmark/hardware"))
    spec = importlib.util.spec_from_file_location(
        "current_accuracy", ROOT / "benchmark/hardware/validate_current_accuracy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_budgets_are_fixed_not_probes_or_multiseed(tool):
    assert tool.SEED == 131
    assert tool.BUDGETS == {"resnet18": (20, "multiseed_v1", 3140),
                            "yolov5s": (300, "multiseed_v2", 600),
                            "detr": (12, "multiseed_v2", 720),
                            "vit_b16": (20, "multiseed_v2", 3140)}


def test_vendor_profile_rejected_before_creating_outputs(tool, tmp_path):
    output = tmp_path / "evidence"
    with pytest.raises(ValueError, match="another vendor"):
        tool.run(Namespace(backend="nvidia", numerical_profile="metax_deterministic_v1", output=output))
    assert not output.exists()


@pytest.mark.parametrize("device", ("0,1", "cuda:0", "-1"))
def test_pair_uses_only_one_explicit_allocated_device(tool, tmp_path, device):
    output = tmp_path / "evidence"
    with pytest.raises(ValueError, match="one actually allocated"):
        tool.run(Namespace(backend="metax", numerical_profile="native", device=device, output=output))
    assert not output.exists()


def test_never_reuses_or_overwrites_an_existing_attempt(tool, tmp_path):
    output = tmp_path / "evidence"
    output.mkdir()
    original = output / "failure.log"
    original.write_text("Preserve the first failure")
    with pytest.raises(FileExistsError):
        tool.run(Namespace(backend="metax", numerical_profile="native", device="0", output=output))
    assert original.read_text() == "Preserve the first failure"
