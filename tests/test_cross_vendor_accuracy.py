import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = load("run_cross_vendor_accuracy")
PLAN_PATH = ROOT / "docs/hardware/validation/cross_vendor_20261021/plan.json"


def report(backend="nvidia"):
    plan = TOOL.load_plan(PLAN_PATH)
    result = {key: "same" for key in plan["pairing"]}
    result.update(status="completed", model="resnet18", seed=131, epochs=20, protocol="multiseed_v1",
                  backend=backend, training_steps=3140, expected_training_steps=3140,
                  losses=[2.] * 3140, learning_rates=[[.05]] * 3140,
                  final_metrics={"top1": 65.}, restored_metrics={"top1": 65.},
                  initial_rng_sha256=backend + "-rng", final_state_sha256=backend + "-weights",
                  nvidia_extension_loaded=backend == "nvidia",
                  environment={"torch": backend + "-torch", "tf32": False,
                               "matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                               "matmul_precision": "highest", "deterministic_algorithms": True})
    return result


def test_plan_preserves_all_four_complete_budgets_and_five_seeds():
    plan = TOOL.load_plan(PLAN_PATH)
    assert set(plan["models"]) == {"resnet18", "yolov5s", "detr", "vit_b16"}
    assert all(spec["seeds"] == [131, 132, 133, 134, 135] for spec in plan["models"].values())
    assert sum(spec["expected_steps_per_run"] * len(spec["seeds"]) * 2
               for spec in plan["models"].values()) == 76000


def test_cross_vendor_pair_allows_different_stacks_rng_and_final_weights():
    result = TOOL.compare_pair(report(), report("metax"), TOOL.load_plan(PLAN_PATH))
    assert result["preconditions_equal"]
    assert not result["environment_equal"]
    assert not result["gpu_rng_equal_diagnostic_only"]
    assert not result["bitwise_final_state_equal"]


@pytest.mark.parametrize("key", TOOL.load_plan(PLAN_PATH)["pairing"])
def test_cross_vendor_pair_rejects_each_changed_precondition(key):
    other = report("metax")
    other[key] = "changed"
    with pytest.raises(ValueError, match=key):
        TOOL.compare_pair(report(), other, TOOL.load_plan(PLAN_PATH))


@pytest.mark.parametrize("key,value", [("status", "probe_completed"), ("training_steps", 2),
                                       ("backend", "metax"), ("losses", [])])
def test_probe_partial_wrong_vendor_and_empty_traces_are_ineligible(key, value):
    item = report()
    item[key] = value
    with pytest.raises(ValueError):
        TOOL.verify_run(item, TOOL.load_plan(PLAN_PATH), "resnet18", 131, "nvidia", "same", "same")


@pytest.mark.parametrize("key,value", [("tf32", True), ("matmul_allow_tf32", True),
                                       ("cudnn_allow_tf32", True), ("matmul_precision", "high"),
                                       ("deterministic_algorithms", False)])
def test_precision_policy_must_be_read_back_not_assumed(key, value):
    item = report()
    item["environment"][key] = value
    with pytest.raises(ValueError):
        TOOL.verify_run(item, TOOL.load_plan(PLAN_PATH), "resnet18", 131, "nvidia", "same", "same")


def test_metax_rejects_nvidia_binary_loading():
    item = report("metax")
    item["nvidia_extension_loaded"] = True
    with pytest.raises(ValueError, match="NVIDIA binary"):
        TOOL.verify_run(item, TOOL.load_plan(PLAN_PATH), "resnet18", 131, "metax", "same", "same")


def test_verified_complete_run_is_eligible():
    TOOL.verify_run(report("metax"), TOOL.load_plan(PLAN_PATH), "resnet18", 131, "metax", "same", "same")


def test_copy_verifies_both_source_and_existing_destination(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"frozen-image")
    digest = TOOL.DATA.copy_verified(source, target)
    assert digest == TOOL.DATA.sha256(source) == TOOL.DATA.sha256(target)
    with pytest.raises(ValueError, match="Source hash"):
        TOOL.DATA.copy_verified(source, target, "incorrect")
    target.write_bytes(b"other-image")
    with pytest.raises(ValueError, match="Refusing to replace"):
        TOOL.DATA.copy_verified(source, target)


def test_duplicate_seeds_are_rejected(tmp_path):
    plan = TOOL.load_plan(PLAN_PATH)
    plan["models"]["resnet18"]["seeds"] = [131, 131]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="distinct"):
        TOOL.load_plan(path)


def test_checkpoint_auditor_checks_nested_tensor_and_scalar_values():
    import torch

    auditor = load("audit_cross_vendor_accuracy")
    count, failures = auditor.finite_tensors({"model": [torch.ones(2), torch.tensor([float("nan")])],
                                            "scheduler": {"lr": float("inf")}})
    assert count == 2
    assert failures == ["state.model.1", "state.scheduler.lr"]


def checkpoint_fixture(directory):
    import torch

    state = {"model": {"weight": torch.ones(2)},
             "optimizer": {"state": {0: {"momentum_buffer": torch.ones(2)}}},
             "scheduler": {"last_epoch": 20}}
    driver = TOOL.sibling("validate_training_accuracy")
    (directory / "report.json").write_text(json.dumps(
        {"status": "completed", "final_state_sha256": driver.state_hash(state["model"])}))
    torch.save(state, directory / "comparison_state.pt")
    return state


def test_audit_verifies_saved_model_and_all_training_sections(tmp_path):
    checkpoint_fixture(tmp_path)
    result = load("audit_cross_vendor_accuracy").audit(tmp_path)
    assert result["status"] == "passed"
    assert result["sections"]["optimizer"]["tensor_count"] == 1
    assert (tmp_path / "checkpoint-audit.json").exists()


@pytest.mark.parametrize("mutation", ["changed_model", "empty_optimizer", "non_finite_optimizer"])
def test_audit_rejects_changed_empty_or_non_finite_state(tmp_path, mutation):
    import torch

    state = checkpoint_fixture(tmp_path)
    if mutation == "changed_model":
        state["model"]["weight"].add_(1)
    elif mutation == "empty_optimizer":
        state["optimizer"].clear()
    else:
        state["optimizer"]["state"][0]["momentum_buffer"][0] = float("nan")
    torch.save(state, tmp_path / "comparison_state.pt")
    with pytest.raises(ValueError):
        load("audit_cross_vendor_accuracy").audit(tmp_path)


def test_evidence_members_must_stay_inside_task_root(tmp_path):
    collector = load("collect_cross_vendor_evidence")
    root = tmp_path / "task"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("not task evidence")
    inside = root / "report.json"
    inside.write_text("{}")
    assert collector.safe_relative(root, inside) == "report.json"
    with pytest.raises(ValueError):
        collector.safe_relative(root, outside)


def test_evidence_cannot_replace_existing_archive(tmp_path):
    archive = tmp_path / "existing.tar.gz"
    archive.write_bytes(b"preserved")
    with pytest.raises(ValueError, match="do not replace"):
        load("collect_cross_vendor_evidence").collect(tmp_path, PLAN_PATH, "nvidia", archive)
    assert archive.read_bytes() == b"preserved"


def test_evidence_requires_every_declared_run_before_packaging(tmp_path):
    root = tmp_path / "task"
    data = root / "data"
    data.mkdir(parents=True)
    (data / "data.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        load("collect_cross_vendor_evidence").collect(root, PLAN_PATH, "nvidia", tmp_path / "evidence.tar.gz")
    assert not (tmp_path / "evidence.tar.gz").exists()


def classification_predictions():
    return [{"filename": "a.png", "label": 1, "prediction": 1, "topk_idx": [1, 2, 3, 4, 5]},
            {"filename": "b.png", "label": 2, "prediction": 1, "topk_idx": [1, 2, 3, 4, 5]}]


def test_independent_classification_counts_match_original_metric_arithmetic(tmp_path):
    meta = tmp_path / "test.txt"
    meta.write_text("a.png 1\nb.png 2\n")
    metrics, counts = load("recompute_cross_vendor_metrics").classification_metrics(classification_predictions(), meta)
    assert metrics == {"top1": 50., "top5": 100.}
    assert counts["correct"] == {1: 1, 5: 2}


@pytest.mark.parametrize("kind", ["missing", "duplicate", "wrong_label", "wrong_top1"])
def test_independent_metrics_reject_invalid_or_incomplete_predictions(tmp_path, kind):
    meta = tmp_path / "test.txt"
    meta.write_text("a.png 1\nb.png 2\n")
    predictions = classification_predictions()
    if kind == "missing":
        predictions.pop()
    elif kind == "duplicate":
        predictions[1]["filename"] = "a.png"
    elif kind == "wrong_label":
        predictions[0]["label"] = 9
    else:
        predictions[0]["prediction"] = 9
    with pytest.raises(ValueError):
        load("recompute_cross_vendor_metrics").classification_metrics(predictions, meta)


def test_independent_detection_uses_official_coco_evaluation(tmp_path):
    pytest.importorskip("pycocotools")
    meta = tmp_path / "test.json"
    meta.write_text(json.dumps({"info": {}, "images": [{"id": 1, "width": 20, "height": 20}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
                         "bbox": [0, 0, 10, 10], "area": 100, "segmentation": []}]}))
    metrics, details = load("recompute_cross_vendor_metrics").detection_metrics(
        [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 1.}], meta)
    assert metrics["bbox.AP"] == pytest.approx(1.)
    assert details["validation_images"] == 1
