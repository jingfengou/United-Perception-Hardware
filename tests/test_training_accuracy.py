import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TOOL = runpy.run_path(str(ROOT / "benchmark/hardware/validate_training_accuracy.py"))
compare, effective_config = TOOL["compare"], TOOL["effective_config"]


def data_manifest():
    return {"cifar10": {"train": {"meta": "/data/cifar/train.txt", "image_dir": "/data/cifar/train"},
                        "test": {"meta": "/data/cifar/test.txt", "image_dir": "/data/cifar/test"}},
            "pennfudan": {"train": {"meta": "/data/penn/train.json", "image_dir": "/data/penn/images"},
                          "test": {"meta": "/data/penn/test.json", "image_dir": "/data/penn/images"}},
            "resnet50_weights": {"path": "/data/resnet50.pth"}}


def test_classification_overlay_uses_real_separate_splits():
    config, path = effective_config(ROOT, data_manifest(), "resnet18", 131, 5)
    assert path == "configs/cls/resnet/res18.yaml"
    assert config["dataset"]["train"]["dataset"]["kwargs"]["meta_file"] != config["dataset"]["test"]["dataset"]["kwargs"]["meta_file"]
    assert config["net"][1]["kwargs"]["num_classes"] == 10
    assert config["net"][2]["kwargs"]["cls_loss"]["kwargs"]["num_classes"] == 10
    assert config["trainer"]["max_epoch"] == 5
    assert config["saver"]["auto_resume"] is False


def test_detection_overlay_preserves_up_model_and_evaluator():
    config, path = effective_config(ROOT, data_manifest(), "faster_rcnn", 132, 3)
    assert path == "configs/det/faster_rcnn/faster_rcnn_r50_fpn_1x.yaml"
    assert config["net"][0]["type"] == "resnet50"
    assert config["dataset"]["test"]["dataset"]["kwargs"]["evaluator"]["kwargs"]["gt_file"] == "/data/penn/test.json"
    assert config["saver"]["pretrain_model"] == "/data/resnet50.pth"
    assert config["net"][-2]["kwargs"]["num_classes"] == 2


def write_fake_run(path, metric=50., weight=1., config_hash="identical"):
    path.mkdir()
    report = {"status": "completed", "model": "resnet18", "seed": 131, "epochs": 1,
              "training_steps": 2, "effective_config_sha256": config_hash,
              "data_manifest_sha256": "data", "initial_state_sha256": "weights",
              "driver_sha256": "driver", "environment": {"torch": "test"},
              "scope": "test fixture only", "losses": [2., 1.], "final_metrics": {"top1": metric}}
    (path / "report.json").write_text(json.dumps(report))
    torch.save({"model": {"weight": torch.tensor([weight])}}, path / "comparison_state.pt")


def comparison_args(tmp_path):
    return argparse.Namespace(compare=[tmp_path / "baseline", tmp_path / "routed"],
                              output=tmp_path / "comparison.json", rtol=1e-4, atol=1e-5, metric_atol=1e-6)


def test_comparison_accepts_equal_results(tmp_path):
    args = comparison_args(tmp_path)
    for path in args.compare:
        write_fake_run(path)
    assert compare(args) == 0


def test_comparison_rejects_metric_regression(tmp_path):
    args = comparison_args(tmp_path)
    write_fake_run(args.compare[0])
    write_fake_run(args.compare[1], metric=49.)
    assert compare(args) == 1


def test_comparison_rejects_weight_regression(tmp_path):
    args = comparison_args(tmp_path)
    write_fake_run(args.compare[0])
    write_fake_run(args.compare[1], weight=1.1)
    assert compare(args) == 1


def test_comparison_rejects_unmatched_experiment(tmp_path):
    args = comparison_args(tmp_path)
    write_fake_run(args.compare[0])
    write_fake_run(args.compare[1], config_hash="different")
    assert compare(args) == 1


def test_legacy_numpy_alias_keeps_original_float64_dtype(monkeypatch):
    import numpy as np
    monkeypatch.setattr(np, "float", None, raising=False)
    TOOL["prepare_legacy_environment"]()
    assert np.float is float
    assert np.array([1, 2], dtype=np.float).dtype == np.dtype("float64")


def test_verified_legacy_weight_conversion_preserves_tensors(tmp_path):
    prepare = runpy.run_path(str(ROOT / "benchmark/hardware/prepare_accuracy_data.py"))
    source, target = tmp_path / "legacy.pth", tmp_path / "modern.pth"
    original = {"layer.weight": torch.arange(6.).reshape(2, 3)}
    torch.save(original, source, _use_new_zipfile_serialization=False)
    prepare["convert_trusted_weights"](source, target)
    restored = torch.load(target, weights_only=True)
    assert torch.equal(restored["layer.weight"], original["layer.weight"])


def test_comparison_rejects_two_incomplete_runs(tmp_path):
    args = comparison_args(tmp_path)
    for path in args.compare:
        write_fake_run(path)
        report = json.loads((path / "report.json").read_text())
        report["status"] = "failed"
        (path / "report.json").write_text(json.dumps(report))
    assert compare(args) == 1


def test_comparison_rejects_empty_training(tmp_path):
    args = comparison_args(tmp_path)
    for path in args.compare:
        write_fake_run(path)
        report = json.loads((path / "report.json").read_text())
        report.update(training_steps=0, losses=[])
        (path / "report.json").write_text(json.dumps(report))
    assert compare(args) == 1


def test_zero_epoch_training_is_rejected_before_gpu_initialization(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "benchmark/hardware/validate_training_accuracy.py"),
                             "--source", str(ROOT), "--data", "unused.json", "--model", "resnet18",
                             "--epochs", "0", "--output", str(tmp_path / "unused")],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "--epochs must be positive" in result.stderr


def test_roi_trace_observes_native_result_without_replacing_it(tmp_path):
    tool = runpy.run_path(str(ROOT / "benchmark/hardware/diagnose_training_divergence.py"))
    trace = tool["RoiTrace"](tmp_path)
    trace.start(0)
    calls = []

    def native(value, output):
        calls.append(True)
        output.copy_(value * 2)

    value, output = torch.tensor([1., 2.]), torch.zeros(2)
    trace.wrap(native)(value, output)
    assert calls == [True]
    assert torch.equal(output, value * 2)
    assert trace.records[0]["native_output"] == trace.records[0]["used_output"]
    saved = torch.load(tmp_path / "roi-0-0.pt", weights_only=True)
    assert torch.equal(saved["gradient"], output)


def test_roi_trace_replay_validates_inputs_and_records_intervention(tmp_path):
    tool = runpy.run_path(str(ROOT / "benchmark/hardware/diagnose_training_divergence.py"))
    reference, output_dir = tmp_path / "reference", tmp_path / "output"
    reference.mkdir()
    output_dir.mkdir()
    value = torch.tensor([1., 2.])
    replayed = torch.tensor([3., 4.])
    torch.save({"inputs": tool["describe"]((value,)), "gradient": replayed}, reference / "roi-0-0.pt")
    trace = tool["RoiTrace"](output_dir, reference)
    trace.start(0)
    calls = []

    def native(value, output):
        calls.append(True)
        output.copy_(value * 2)

    output = torch.zeros(2)
    trace.wrap(native)(value, output)
    assert calls == [True]
    assert torch.equal(output, replayed)
    assert trace.records[0]["native_output"] != trace.records[0]["used_output"]


def test_roi_trace_replay_rejects_mismatched_inputs(tmp_path):
    tool = runpy.run_path(str(ROOT / "benchmark/hardware/diagnose_training_divergence.py"))
    reference, output_dir = tmp_path / "reference", tmp_path / "output"
    reference.mkdir()
    output_dir.mkdir()
    torch.save({"inputs": tool["describe"]((torch.ones(2),)), "gradient": torch.ones(2)},
               reference / "roi-0-0.pt")
    trace = tool["RoiTrace"](output_dir, reference)
    trace.start(0)
    with pytest.raises(AssertionError, match="Replay inputs differ"):
        trace.wrap(lambda value, output: output.copy_(value))(torch.zeros(2), torch.zeros(2))
    assert trace.records == []
    assert not (output_dir / "roi-0-0.pt").exists()


def test_diagnostic_tensor_descriptors_preserve_shape_and_dtype():
    tool = runpy.run_path(str(ROOT / "benchmark/hardware/diagnose_training_divergence.py"))
    describe = tool["describe"]
    assert describe(torch.ones(2)) != describe(torch.ones(1, 2))
    assert describe(torch.ones(2)) != describe(torch.ones(2, dtype=torch.float64))
