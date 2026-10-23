"""Validation tooling must detect mismatches rather than hide failed GPU runs."""
import copy
import importlib.util
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("platform_equivalence", ROOT / "benchmark/hardware/validate_platform_equivalence.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_native_is_the_unchanged_default_profile():
    assert tool.numerical_profile(Namespace(backend="nvidia")) == "native"
    assert (tool.SEED, tool.STEPS, tool.RTOL, tool.ATOL) == (131, 3, 1e-4, 1e-5)


@pytest.mark.parametrize("profile", tool.PROFILES)
def test_metax_profile_is_explicit_and_not_substituted(profile):
    assert tool.numerical_profile(Namespace(backend="metax", numerical_profile=profile)) == profile


@pytest.mark.parametrize("profile", tool.PROFILES[1:])
def test_no_metax_profile_on_nvidia_or_created_failure_directory(tmp_path, profile):
    output = tmp_path / "attempt"
    with pytest.raises(ValueError, match="another vendor"):
        tool.suite(Namespace(backend="nvidia", numerical_profile=profile, output=output))
    assert not output.exists()


def test_unknown_numerical_profile_cannot_silently_become_native():
    with pytest.raises(ValueError, match="Unknown numerical profile"):
        tool.numerical_profile(Namespace(backend="metax", numerical_profile="unknown"))


def test_nested_exact_comparison_and_empty_tensors():
    values = {"nested": [torch.ones(2), None, {"indices": torch.empty(0, dtype=torch.long)}]}
    result = tool.close_trees(values, copy.deepcopy(values))
    assert result == {"tensor_count": 2, "exact_tensor_count": 2, "max_abs": 0., "max_rel": 0.}
    assert tool.tree_hash(values) == tool.tree_hash(copy.deepcopy(values))


@pytest.mark.parametrize("value", [torch.tensor([float("nan")]), torch.tensor([float("inf")])])
def test_comparison_rejects_nonfinite_values(value):
    with pytest.raises(AssertionError):
        tool.close_trees(value, value.clone())


def test_comparison_rejects_shape_dtype_missing_keys_and_changed_values():
    expected = {"prediction": torch.ones(2)}
    for actual in ({}, {"prediction": torch.ones(1)}, {"prediction": torch.ones(2, dtype=torch.float64)},
                   {"prediction": torch.zeros(2)}):
        with pytest.raises(AssertionError):
            tool.close_trees(expected, actual)


def test_source_manifest_is_line_ending_independent(tmp_path):
    source = tmp_path / "up"
    source.mkdir()
    path = source / "example.py"
    path.write_bytes(b"pass\r\n")
    first = tool.source_manifest(tmp_path)
    path.write_bytes(b"pass\n")
    assert tool.source_manifest(tmp_path) == first
    path.write_bytes(b"raise RuntimeError\n")
    assert tool.source_manifest(tmp_path) != first


def test_deployment_checkpoint_is_tensor_only_without_training_objects(tmp_path):
    import numpy as np
    state = {"model": {"layer.weight": torch.randn(2, 3)}, "scheduler": {"lr": np.float64(.1)}}
    original = tool.tree_hash(state)
    path = tmp_path / "weights.pt"
    tool.save_inference_weights(state, path)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    assert restored.keys() == {"model"}
    assert torch.equal(restored["model"]["layer.weight"], state["model"]["layer.weight"])
    assert tool.tree_hash(state) == original


def test_subset_manifest_hashes_describe_selected_images_not_parent(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmark/hardware"))
    from prepare_cross_vendor_data import verify_data
    data = {}
    for dataset in ("cifar10", "pennfudan"):
        data[dataset] = {}
        for split, count in (("train", 16), ("test", 8)):
            folder = tmp_path / dataset / split
            folder.mkdir(parents=True)
            records, images = [], []
            for i in range(count):
                name = "{}.png".format(i)
                (folder / name).write_bytes(str(i).encode())
                if dataset == "cifar10":
                    records.append([name, tool.file_hash(folder / name), i % 10])
                else:
                    mask = folder / (str(i) + "_mask.png")
                    mask.write_bytes(b"mask")
                    records.append([name, tool.file_hash(folder / name), tool.file_hash(mask)])
                    images.append({"id": i, "file_name": name})
            meta = folder / ("meta.txt" if dataset == "cifar10" else "meta.json")
            if dataset == "cifar10":
                meta.write_text("".join("{} {}\n".format(name, label) for name, _, label in records))
            else:
                meta.write_text(json.dumps({"images": images, "annotations": [
                    {"image_id": image["id"], "id": image["id"]} for image in images]}))
            data[dataset][split] = {"meta": str(meta), "count": count, "indices": list(range(count)),
                                    "image_dir": str(folder), "mask_dir": str(folder),
                                    "meta_sha256": tool.file_hash(meta), "image_manifest_sha256": hashlib.sha256(
                                        json.dumps(records, sort_keys=True).encode()).hexdigest()}
    asset = tmp_path / "weights.bin"
    asset.write_bytes(b"asset")
    external = tmp_path / "external"
    external.mkdir()
    (external / "model.py").write_bytes(b"pass\n")
    data["detr"] = {"source": str(external), "checkpoint": str(asset), "checkpoint_sha256": tool.file_hash(asset),
                    "source_sha256": hashlib.sha256(json.dumps(
                        {"model.py": tool.file_hash(external / "model.py")}, sort_keys=True).encode()).hexdigest()}
    for name in ("yolov5_anchors", "resnet50_weights"):
        data[name] = {"path": str(asset), "sha256": tool.file_hash(asset)}
    source = tmp_path / "original.json"
    source.write_text(json.dumps(data))
    original = tool.file_hash(source)
    output = tool.prepare_subset(source, tmp_path / "selected")
    selected = json.loads(output.read_text())
    verify_data(selected)
    assert tool.file_hash(source) == original
    for dataset in ("cifar10", "pennfudan"):
        for split, count in (("train", 8), ("test", 4)):
            assert selected[dataset][split]["count"] == len(selected[dataset][split]["indices"]) == count
            assert selected[dataset][split]["image_manifest_sha256"] != data[dataset][split]["image_manifest_sha256"]
    with pytest.raises(FileExistsError):
        tool.prepare_subset(source, tmp_path / "selected")
