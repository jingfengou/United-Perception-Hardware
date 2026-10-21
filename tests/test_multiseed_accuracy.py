import copy
import json
from pathlib import Path
import runpy

import pytest
from scipy import stats
import torch

ROOT = Path(__file__).resolve().parents[1]
SUITE = runpy.run_path(str(ROOT / "tools/run_multiseed_accuracy.py"))
DRIVER = runpy.run_path(str(ROOT / "tools/validate_training_accuracy.py"))
ADAPTER = runpy.run_path(str(ROOT / "tools/accuracy_detr_adapter.py"))
ASSETS = runpy.run_path(str(ROOT / "tools/prepare_multiseed_extension.py"))
SMOKE = runpy.run_path(str(ROOT / "tools/validate_models.py"))


def test_tost_matches_independent_scipy_one_sided_tests():
    values = [-.4, .2, -.1, .3, .1]
    result = SUITE["equivalent"](values, 1.)
    lower = stats.ttest_1samp(values, -1., alternative="greater").pvalue
    upper = stats.ttest_1samp(values, 1., alternative="less").pvalue
    assert result["tost_p_lower"] == pytest.approx(lower)
    assert result["tost_p_upper"] == pytest.approx(upper)
    assert result["equivalent_within_margin"] is True
    assert -1 < result["confidence_interval_90"][0] < result["confidence_interval_90"][1] < 1


def test_zero_mean_with_high_variance_does_not_prove_equivalence():
    result = SUITE["equivalent"]([-10., 10., -10., 10.], 1.)
    assert result["mean_delta_percentage_points"] == 0
    assert result["equivalent_within_margin"] is False


@pytest.mark.parametrize("delta,expected", [(0., True), (.5, True), (1., False), (-1., False), (-2., False)])
def test_zero_variance_and_margin_boundaries(delta, expected):
    assert SUITE["equivalent"]([delta] * 5, 1.)["equivalent_within_margin"] is expected


@pytest.mark.parametrize("values,margin", [([0.], 1.), ([0., float("nan")], 1.), ([0., 0.], 0.)])
def test_equivalence_rejects_invalid_input(values, margin):
    with pytest.raises(ValueError):
        SUITE["equivalent"](values, margin)


def example_report():
    return {"status": "completed", "seed": 131, "epochs": 12, "protocol": "multiseed_v1",
            "data_manifest_sha256": "data", "driver_sha256": "driver", "training_steps": 2,
            "expected_training_steps": 2, "losses": [2., 1.], "learning_rates": [[.1], [.01]],
            "final_metrics": {"bbox.AP": .5}, "restored_metrics": {"bbox.AP": .5}}


@pytest.mark.parametrize("key,value", [("status", "failed"), ("training_steps", 1),
                                       ("losses", []), ("learning_rates", []),
                                       ("data_manifest_sha256", "different"),
                                       ("driver_sha256", "different"),
                                       ("final_metrics", {"bbox.AP": float("nan")})])
def test_suite_rejects_incomplete_or_changed_runs(key, value):
    report = example_report()
    report[key] = value
    with pytest.raises(ValueError):
        SUITE["verify_report"](report, {"epochs": 12, "primary_metric": "bbox.AP"},
                               131, "data", "driver", "multiseed_v1")


def test_valid_run_has_complete_budget_and_restore():
    SUITE["verify_report"](example_report(), {"epochs": 12, "primary_metric": "bbox.AP"},
                           131, "data", "driver", "multiseed_v1")


def test_predeclared_plan_has_distinct_seeds_and_external_detr_label():
    plan = SUITE["load_plan"](ROOT / "docs/hardware/validation/multiseed_20261021/plan.json")
    assert len(plan["models"]["faster_rcnn"]["seeds"]) == 10
    assert plan["models"]["faster_rcnn"]["repeat_seeds"] == [131, 132, 133]
    assert plan["equivalence"]["margin_percentage_points"] == 1.
    assert "NOT an upstream UP model" in plan["models"]["detr"]["provenance"]


def test_duplicate_seeds_are_not_independent_evidence(tmp_path):
    plan = json.loads((ROOT / "docs/hardware/validation/multiseed_20261021/plan.json").read_text())
    plan["models"]["resnet18"]["seeds"] = [131, 131]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="distinct"):
        SUITE["load_plan"](path)


def test_scaled_up_inclusive_targets_round_trip_to_detr_and_back():
    info = [200, 150, (2., 1.5), 100, 100, False, 0, 0]
    scaled_up = torch.tensor([[15., 40., 43.5, 118.]])
    normalized = ADAPTER["target_boxes"](scaled_up, info, 1)
    assert torch.allclose(normalized, torch.tensor([[.2, .4, .2, .4]]))
    half_open = torch.tensor([[15., 40., 45., 120.]])
    restored = ADAPTER["up_prediction_boxes"](half_open, info, 1)
    assert torch.equal(restored, scaled_up)
    original = restored / restored.new_tensor([1.5, 2., 1.5, 2.])
    assert torch.equal(original[:, 2:] - original[:, :2] + 1, torch.tensor([[20., 40.]]))


def test_adapter_keeps_empty_target_contract():
    empty = torch.empty(0, 4)
    result = ADAPTER["target_boxes"](empty, [64, 96, 1., 64, 96, False, 0, 0], 1)
    assert result.shape == (0, 4)


def test_source_pin_detects_external_python_changes(tmp_path):
    path = tmp_path / "model.py"
    path.write_text("value = 1\n")
    before = ADAPTER["source_hash"](tmp_path)
    path.write_text("value = 2\n")
    assert ADAPTER["source_hash"](tmp_path) != before


def test_batch_hash_covers_labels_and_does_not_consume_rng():
    batch = {"image": torch.zeros(2, 3), "labels": torch.tensor([0, 1])}
    before = torch.get_rng_state().clone()
    first = DRIVER["batch_hash"](batch)
    assert torch.equal(before, torch.get_rng_state())
    changed = copy.deepcopy(batch)
    changed["labels"][0] = 1
    assert DRIVER["batch_hash"](changed) != first


def configuration(model, epochs, protocol="multiseed_v1"):
    data = {"cifar10": {split: {"meta": "/data/cifar/" + split, "image_dir": "/data/cifar"}
                         for split in ("train", "test")},
            "pennfudan": {split: {"meta": "/data/penn/" + split, "image_dir": "/data/penn"}
                          for split in ("train", "test")},
            "resnet50_weights": {"path": "/weights/resnet50.pth"},
            "yolov5_anchors": {"path": "/weights/yolov5-anchors.json", "sha256": "anchors"},
            "detr": {"source": "/official-detr", "checkpoint": "/weights/detr.pt",
                     "source_sha256": "source", "checkpoint_sha256": "weights"}}
    return DRIVER["effective_config"](ROOT, data, model, 131, epochs, protocol)[0]


def test_full_faster_rcnn_schedule_keeps_original_batch_scaling():
    config = configuration("faster_rcnn", 12)
    scheduler = config["trainer"]["lr_scheduler"]
    assert scheduler["warmup_iter"] == 60
    assert scheduler["kwargs"]["milestones"] == [8, 11]
    assert scheduler.get("warmup_register_type", "base") == "base"


def test_full_classification_schedule_retains_no_scale_lr():
    config = configuration("resnet18", 20)
    scheduler = config["trainer"]["lr_scheduler"]
    assert scheduler["warmup_iter"] == 0
    assert scheduler["kwargs"]["milestones"] == [12, 18]
    assert scheduler["warmup_register_type"] == "no_scale_lr"


def test_detr_configuration_uses_external_adapter_and_actual_grad_clipper():
    config = configuration("detr", 12)
    assert config["net"][0]["type"] == "accuracy_detr_adapter.DetrForUP"
    assert config["trainer"]["optimizer"]["type"] == "AdamW"
    assert "pretrain_model" not in config["saver"]
    assert config["hooks"] == [{"type": "grad_clipper", "kwargs": {
        "mode": "pre_defined", "max_norm": .1, "norm_type": 2}}]


def test_model_mismatch_is_not_reused_as_another_model():
    report = dict(example_report(), model="faster_rcnn")
    with pytest.raises(ValueError, match="different model"):
        SUITE["verify_report"](report, {"epochs": 12, "primary_metric": "bbox.AP"},
                               131, "data", "driver", "multiseed_v1", "detr")


@pytest.mark.parametrize("key", ["data_manifest_sha256", "source_manifest_sha256", "driver_sha256"])
def test_cross_seed_source_or_data_changes_are_rejected(key):
    report = dict(example_report(), source_manifest_sha256="source",
                  native_binaries={"extension.so": "hash"}, environment={"torch": "same", "visible_devices": "2"})
    cohorts = {}
    SUITE["verify_cohort"](cohorts, "baseline", report)
    changed = copy.deepcopy(report)
    changed[key] = "different"
    with pytest.raises(ValueError, match="Mixed source"):
        SUITE["verify_cohort"](cohorts, "baseline", changed)


def test_low_memory_active_compute_process_is_still_rejected(monkeypatch):
    monkeypatch.setattr(SUITE["subprocess"], "check_output", lambda *args, **kwargs: "12345\n")
    with pytest.raises(RuntimeError, match="active compute"):
        SUITE["ensure_device_idle"]("2")


def test_idle_gpu_admission_checks_compute_processes_and_memory(monkeypatch):
    values = iter(["", "2\n"])
    monkeypatch.setattr(SUITE["subprocess"], "check_output", lambda *args, **kwargs: next(values))
    SUITE["ensure_device_idle"]("2")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_flat_label_cross_entropy_preserves_weighted_loss_and_gradient(dtype):
    generator = torch.Generator().manual_seed(131)
    logits = torch.randn(2, 100, 2, generator=generator, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, 2, (2, 100), generator=generator)
    weights = torch.tensor([1., .1], dtype=dtype)
    original = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, weights)
    flattened = ADAPTER["label_cross_entropy"](logits, labels, weights)
    expected = torch.autograd.grad(original, logits, retain_graph=True)[0]
    actual = torch.autograd.grad(flattened, logits)[0]
    torch.testing.assert_close(original, flattened)
    torch.testing.assert_close(expected, actual)


def test_extension_plan_preserves_original_plan_and_requires_five_distinct_seeds():
    plan = SUITE["load_plan"](ROOT / "docs/hardware/validation/multiseed_20261021/extension-plan.json")
    assert plan["protocol"] == "multiseed_v2"
    assert set(plan["models"]) == {"yolov5s", "vit_b16", "detr"}
    for spec in plan["models"].values():
        assert len(set(spec["seeds"])) == 5


def test_native_yolo_recipe_keeps_architecture_and_uses_shared_pinned_anchors():
    config = configuration("yolov5s", 300, "multiseed_v2")
    assert [module["type"] for module in config["net"]] == [
        "DarkNetv5", "DarknetPAFPN", "YoloV5Head", "yolov5_post"]
    assert config["net"][-1]["kwargs"]["num_classes"] == 2
    assert config["net"][-1]["kwargs"]["cfg"]["anchor_generator"]["kwargs"]["base_anchors_file"] == "/weights/yolov5-anchors.json"
    transforms = config["dataset"]["train"]["dataset"]["kwargs"]["transformer"]
    assert transforms[0]["kwargs"]["tar_size"] == 320
    assert transforms[1]["kwargs"]["border"] == [-160, -160]
    assert config["dataset"]["batch_sampler"]["kwargs"]["batch_size"] == 64
    assert config["trainer"]["lr_scheduler"]["warmup_iter"] == 0
    assert config["trainer"]["lr_scheduler"]["warmup_epochs"] == 3
    assert config["trainer"]["lr_scheduler"]["kwargs"]["T_max"] == 300
    assert config["ema"]["enable"] is False


def test_vit_keeps_native_base_model_and_updates_every_class_count():
    config = configuration("vit_b16", 20, "multiseed_v2")
    assert config["net"][0]["type"] == "vit_base_patch16_224"
    assert config["net"][0]["kwargs"]["image_size"] == 64
    assert config["net"][1]["kwargs"]["in_plane"] == 768
    assert config["net"][1]["kwargs"]["num_classes"] == 10
    assert config["net"][2]["kwargs"]["cls_loss"]["kwargs"]["num_classes"] == 10
    batch = config["dataset"]["train"]["dataloader"]["kwargs"]["batch_fn"]
    assert batch["type"] == "batch_cutmixup"
    assert batch["kwargs"]["num_classes"] == 10
    assert config["trainer"]["lr_scheduler"]["warmup_register_type"] == "no_scale_lr"
    assert config["trainer"]["lr_scheduler"]["warmup_iter"] == 157


def test_corrected_detr_recipe_does_not_scale_or_erase_backbone_learning_rate():
    config = configuration("detr", 12, "multiseed_v2")
    optimizer = config["trainer"]["optimizer"]
    assert optimizer["kwargs"]["lr"] == 1e-4
    assert optimizer["special_param_group"] == [{"key": "detector.backbone", "lr": 1e-5}]
    assert config["trainer"]["lr_scheduler"]["warmup_register_type"] == "no_scale_lr"


@pytest.mark.parametrize("model", ["yolov5s", "vit_b16"])
def test_additional_models_do_not_silently_reuse_an_old_recipe(model):
    with pytest.raises(ValueError, match="predeclared"):
        configuration(model, 20)


def test_completed_run_still_must_match_the_predeclared_number_of_steps():
    with pytest.raises(ValueError, match="predeclared"):
        SUITE["verify_report"](example_report(), {
            "epochs": 12, "primary_metric": "bbox.AP", "expected_steps_per_run": 720},
            131, "data", "driver", "multiseed_v1")


def test_anchor_preparation_is_repeatable_and_refuses_different_assets(tmp_path):
    manifest = tmp_path / "data.json"
    manifest.write_text(json.dumps({"pennfudan": {"train": "fixed"}}))
    output = tmp_path / "new" / "data.json"
    first = ASSETS["add_anchors"](ROOT, manifest, output)
    assert first == ASSETS["add_anchors"](ROOT, manifest, output)
    assert json.loads(output.read_text())["pennfudan"] == {"train": "fixed"}
    Path(first["path"]).write_text("[]")
    with pytest.raises(ValueError, match="different anchor"):
        ASSETS["add_anchors"](ROOT, manifest, output)


@pytest.mark.parametrize("name,size", [("yolov5s", 640), ("vit_b16", 224)])
def test_additional_native_shape_smoke_does_not_shrink_inputs(name, size):
    batch = SMOKE["image_input"](name, "cpu")
    assert batch["image"].shape == (2, 3, size, size)
    target = "gt" if name == "vit_b16" else "gt_bboxes"
    assert target in batch
