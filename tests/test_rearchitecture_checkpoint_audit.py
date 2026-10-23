"""Audit gates must reject incomplete runs and keep the frozen metric tolerance."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tool(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmark/hardware"))
    spec = importlib.util.spec_from_file_location(
        "checkpoint_audit", ROOT / "benchmark/hardware/audit_rearchitecture_checkpoints.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metric_comparison_uses_the_existing_absolute_tolerance(tool):
    tool.compare_metrics({"AP": .7}, {"AP": .7 + 5e-7})
    with pytest.raises(AssertionError, match="metric changed"):
        tool.compare_metrics({"AP": .7}, {"AP": .7 + 2e-6})


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_metrics_cannot_pass_with_nonfinite_values(tool, value):
    with pytest.raises(AssertionError, match="Non-finite"):
        tool.compare_metrics({"AP": value}, {"AP": value})


def test_missing_metrics_fail(tool):
    with pytest.raises(AssertionError, match="keys changed"):
        tool.compare_metrics({"AP": .7}, {})


@pytest.mark.parametrize("correct_tree", (True, False))
def test_selected_platform_must_come_from_the_requested_frozen_tree(tool, tmp_path, monkeypatch, correct_tree):
    root = tmp_path / "selected"
    module = root / "packages/up-platform/src/up_platform/__init__.py"
    actual = module if correct_tree else tmp_path / "site-packages/up_platform/__init__.py"
    monkeypatch.setitem(sys.modules, "up_platform", SimpleNamespace(__file__=str(actual)))
    args = SimpleNamespace(source=root)
    if correct_tree:
        assert tool.platform_source(args) == str(module.resolve())
    else:
        with pytest.raises(RuntimeError, match="different source tree"):
            tool.platform_source(args)


@pytest.mark.parametrize("component", ("model", "optimizer", "scheduler", "gradients", "routing"))
def test_eval_snapshot_detects_each_protected_state_change(tool, component):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    runner = SimpleNamespace(model=model, optimizer=optimizer,
                             lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 1))
    owner = ["portable"]
    accelerator = SimpleNamespace(capabilities=lambda: {
        "purpose": "training", "backend": "test",
        "operations": {"test": {"owner": owner[0], "scope": "common"}}})
    before = tool.state_snapshot(runner, accelerator, True)
    if component == "model":
        with torch.no_grad():
            model.weight.add_(1)
    elif component == "optimizer":
        optimizer.param_groups[0]["lr"] = .2
    elif component == "scheduler":
        runner.lr_scheduler.last_epoch += 1
    elif component == "gradients":
        model.weight.grad = torch.ones_like(model.weight)
    else:
        owner[0] = "different"
    assert before != tool.state_snapshot(runner, accelerator, True)


@pytest.mark.parametrize("model", ("resnet18", "yolov5s", "detr", "vit_b16"))
def test_audit_accepts_only_the_complete_predeclared_run(tool, tmp_path, model):
    epochs, protocol, steps = tool.BUDGETS[model]
    report = {"status": "completed", "model": model, "backend": "metax", "seed": 131,
              "epochs": epochs, "protocol": protocol, "training_steps": steps,
              "environment": {"numerical_profile": "metax_deterministic_v2"}}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    assert tool.complete_training_report(tmp_path, model, "metax", "metax_deterministic_v2") == report
    for key, value in (("training_steps", steps - 1), ("status", "probe_completed"), ("seed", 132)):
        path.write_text(json.dumps(dict(report, **{key: value})))
        with pytest.raises(ValueError, match="complete predeclared"):
            tool.complete_training_report(tmp_path, model, "metax", "metax_deterministic_v2")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="profile"):
        tool.complete_training_report(tmp_path, model, "metax", "native")
    environment = {"precision": {"profile": "metax_deterministic_v2"}, "visible_devices": "2"}
    reference = {"passed": True, "action": "training", "model": model, "backend": "metax",
                 "environment": environment}
    args = SimpleNamespace(model=model, backend="metax")
    tool.validate_reference(reference, args, environment)
    for key, value in (("passed", False), ("action", "inference"), ("model", "different"),
                       ("backend", "nvidia"), ("environment", dict(environment, visible_devices="3"))):
        with pytest.raises(ValueError, match="Reference"):
            tool.validate_reference(dict(reference, **{key: value}), args, environment)
