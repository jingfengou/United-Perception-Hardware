"""Contracts for the explicitly selected vendor model validation path."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[1] / "tools/validate_models.py"
    spec = importlib.util.spec_from_file_location("vendor_model_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("nested", [False, True])
def test_detr_manifest_preserves_pinned_inputs(driver, tmp_path, nested):
    entry = {"source": "/external/detr", "checkpoint": "/weights/detr.pt",
             "source_sha256": "source-hash", "checkpoint_sha256": "weights-hash"}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"detr": entry} if nested else entry))
    config = driver.prepare_detr_config(path)
    assert config["num_classes"] == 2
    assert config["net"][0]["type"] == "accuracy_detr_adapter.DetrForUP"
    assert config["net"][0]["kwargs"] == entry


@pytest.mark.parametrize("entry", [{}, {"source": ""}, {"source": 1}])
def test_detr_manifest_rejects_unpinned_inputs(driver, tmp_path, entry):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(entry))
    with pytest.raises(ValueError, match="pinned source"):
        driver.prepare_detr_config(path)


def test_detr_synthetic_batch_matches_adapter_contract(driver):
    batch = driver.image_input("detr", torch.device("cpu"))
    assert batch["image"].shape == (2, 3, 128, 128)
    assert batch["image"].device.type == "cpu"
    assert all(len(info) == 8 and info[6:] == [0, 0] for info in batch["image_info"])
    assert len(batch["gt_bboxes"]) == 2
    assert all(boxes.shape == (1, 5) for boxes in batch["gt_bboxes"])


def test_default_precision_is_observed_not_changed(driver):
    before = torch.get_float32_matmul_precision()
    report = driver.configure_validation_precision(False, "cpu")
    assert report == {"strict_fp32_requested": False, "float32_matmul_precision": before}
    assert torch.get_float32_matmul_precision() == before


def test_strict_cpu_precision_is_explicit(driver):
    before = torch.get_float32_matmul_precision()
    try:
        report = driver.configure_validation_precision(True, "cpu")
        assert report["strict_fp32_requested"]
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_float32_matmul_precision(before)


def test_strict_precision_rejects_unverified_device_controls(driver):
    with pytest.raises(ValueError, match="only verified"):
        driver.configure_validation_precision(True, "vendor_device")
