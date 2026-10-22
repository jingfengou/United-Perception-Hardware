"""Standalone tests: this directory must be runnable without installing UP."""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import up_platform as platform
from up_platform import (BackendSpec, CapabilityError, DeviceBackend, configure_platform,
                         register_backend, register_op)
from up_platform.registry import BACKENDS, BACKEND_PROBES, OPERATIONS


@pytest.fixture(autouse=True)
def isolate():
    old = dict(BACKENDS), copy.deepcopy(OPERATIONS), dict(BACKEND_PROBES)
    configure_platform({"backend": "cpu"}, purpose="training", reset=True)
    yield
    BACKENDS.clear()
    BACKENDS.update(old[0])
    OPERATIONS.clear()
    OPERATIONS.update(old[1])
    BACKEND_PROBES.clear()
    BACKEND_PROBES.update(old[2])
    configure_platform({"backend": "cpu"}, purpose="training", reset=True)


def spec(name="test_vendor", device_factory=DeviceBackend, training=True, inference=True):
    return BackendSpec(name, device_factory.device_type, device_factory,
                       "up_platform.training.backend:TrainingBackend" if training else None,
                       "up_platform.inference.backend:InferenceBackend" if inference else None)


def test_import_needs_neither_torch_nor_up(tmp_path):
    code = """
import sys
class Guard:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('up', 'torch', 'torch_npu', 'torch_mlu', 'torchvision', 'triton'):
            raise AssertionError('unexpected import: ' + fullname)
sys.meta_path.insert(0, Guard())
import up_platform
assert len(up_platform.registry.OPERATIONS) > 73
assert 'up' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)


@pytest.mark.parametrize("purpose,forbidden", [("training", "inference"), ("inference", "training")])
def test_purpose_import_isolation(purpose, forbidden, tmp_path):
    code = """
import sys
import up_platform
class Guard:
    def find_spec(self, fullname, *args):
        if '{forbidden}' in fullname.split('.') or fullname.split('.')[0] in ('torch_npu', 'torch_mlu', 'up'):
            raise AssertionError('unexpected import: ' + fullname)
sys.meta_path.insert(0, Guard())
ctx = up_platform.configure_platform({{'backend': 'cpu'}}, purpose='{purpose}')
with ctx.autocast(enabled=False):
    pass
assert ctx.purpose == '{purpose}'
""".format(purpose=purpose, forbidden=forbidden)
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)


def test_distinct_namespace_and_single_override():
    class DifferentDevice(DeviceBackend):
        device_type = "privateuseone"
    register_backend(spec(device_factory=DifferentDevice))
    register_op("test_vendor", "gpu_iou_overlap", lambda a, b: (a, b, "override"), scope="common")
    context = configure_platform({"backend": "test_vendor", "device": "privateuseone:1"},
                                 purpose="training", reset=True)
    assert str(context.device) == "privateuseone:1"
    assert context.call("gpu_iou_overlap", 1, 2) == (1, 2, "override")
    dets = torch.tensor([[0., 0., 2., 2., .8], [0., 0., 2., 2., .7]])
    assert context.ops.naive_nms(dets, .5).tolist() == [0]
    assert context.capabilities()["operations"]["naive_nms"]["owner"] == "portable"
    # CPU inputs here exercise routing only, not execution on privateuseone hardware.


def test_scope_does_not_follow_eval_no_grad_or_requires_grad():
    register_backend(spec())
    register_op("test_vendor", "marker", lambda x: "train", scope="training")
    register_op("test_vendor", "marker", lambda x: "deploy", scope="inference")
    context = configure_platform({"backend": "test_vendor"}, purpose="training", reset=True)
    model = torch.nn.Linear(2, 2).eval()
    with torch.no_grad():
        assert context.call("marker", model(torch.ones(1, 2))) == "train"
    assert context.purpose == "training"
    with pytest.raises(AttributeError):
        context.purpose = "inference"
    with pytest.raises(platform.BackendConfigurationError):
        configure_platform({"backend": "test_vendor"}, purpose="inference")
    context = configure_platform({"backend": "test_vendor"}, purpose="inference", reset=True)
    assert context.call("marker", torch.ones(1, requires_grad=True)) == "deploy"


def test_same_purpose_then_common_never_other_purpose():
    register_backend(spec())
    register_op("test_vendor", "marker", lambda: "common", scope="common")
    register_op("test_vendor", "marker", lambda: "train", scope="training")
    register_op("test_vendor", "inference_only", lambda: 1, scope="inference")
    context = configure_platform({"backend": "test_vendor"}, purpose="training", reset=True)
    assert context.call("marker") == "train"
    with pytest.raises(CapabilityError, match="test_vendor.*training.*inference_only"):
        context.call("inference_only")
    context = configure_platform({"backend": "test_vendor"}, purpose="inference", reset=True)
    assert context.call("marker") == "common"


def test_inference_only_backend_and_training_helpers():
    register_backend(spec(training=False))
    with pytest.raises(CapabilityError, match="no execution backend"):
        configure_platform({"backend": "test_vendor"}, purpose="training", reset=True)
    context = configure_platform({"backend": "test_vendor"}, purpose="inference", reset=True)
    for method in ("grad_scaler", "wrap_model", "build_fused_optimizer", "reload_fused_optimizer"):
        with pytest.raises(CapabilityError, match="requires training"):
            getattr(context, method)
    with pytest.raises(CapabilityError, match="distributed.wrap_model"):
        context.call("distributed.wrap_model", torch.nn.Linear(2, 2))


def test_dtype_variant_dependencies_and_execution_status():
    register_op("cpu", "test", lambda x, mode="x": x * 2, scope="training",
                dtypes=(torch.float32,), variants={"mode": ("x",)}, dependencies=("missing_platform_test_sdk",),
                backward=True)
    context = configure_platform({"backend": "cpu"}, purpose="training", reset=True)
    info = context.capabilities()["operations"]["test"]
    assert info["registered"] and info["dependencies_available"] is None
    assert info["executed"] == 0 and not info["hardware_validated"]
    with pytest.raises(CapabilityError, match="float64"):
        context.call("test", torch.ones(2, dtype=torch.float64))
    with pytest.raises(CapabilityError, match="unsupported variant"):
        context.call("test", torch.ones(2), "bad")
    with pytest.raises(CapabilityError, match="missing_platform_test_sdk"):
        context.require("test", check_dependencies=True)
    value = torch.ones(2, requires_grad=True)
    context.call("test", value).sum().backward()
    assert value.grad.tolist() == [2., 2.]
    assert context.capabilities()["operations"]["test"]["executed"] == 1


def test_context_freezes_registration_after_initialization():
    register_op("cpu", "marker", lambda: 1, scope="common")
    context = configure_platform({"backend": "cpu"}, purpose="training", reset=True)
    register_op("cpu", "marker", lambda: 2, scope="common", replace=True)
    assert context.call("marker") == 1
    context.runtime.op_fallbacks = ("undeclared_provider",)
    assert context.ops.marker() == 1
    with pytest.raises(CapabilityError, match="complete backward"):
        context.require("marker", backward=True)
    assert configure_platform({"backend": "cpu"}, purpose="training", reset=True).call("marker") == 2


def test_vendor_environment_constraints_cannot_be_overwritten():
    register_backend(BackendSpec("required_env", "cpu", DeviceBackend,
                                "up_platform.training.backend:TrainingBackend", requirements={"python": "<2"}))
    with pytest.raises(platform.BackendConfigurationError, match="requires python"):
        configure_platform({"backend": "required_env", "requirements": {"python": ">=3"}},
                           purpose="training", reset=True)


def test_profile_switch_requires_reset():
    with pytest.raises(platform.BackendConfigurationError, match="Reset the platform"):
        configure_platform({"backend": "cpu", "numerics": "portable_v1"}, purpose="training")


def test_lazily_selected_factory_can_register_its_operations(tmp_path, monkeypatch):
    (tmp_path / "lazy_platform_example.py").write_text(
        'from up_platform import register_op\n'
        'from up_platform.training.backend import TrainingBackend\n'
        'register_op("lazy_vendor", "marker", lambda: 42, scope="training")\n'
        'class Backend(TrainingBackend):\n    pass\n', encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    register_backend(BackendSpec("lazy_vendor", "cpu", DeviceBackend, "lazy_platform_example:Backend"))
    context = configure_platform({"backend": "lazy_vendor"}, purpose="training", reset=True)
    assert context.call("marker") == 42


def test_training_eval_restores_state_and_does_not_update_training_objects():
    from up_platform.training.evaluation import evaluation_context
    context = platform.get_platform()
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3), torch.nn.Dropout(.2))
    model.train()
    model[1].eval()
    model[0].bias.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    scaler = context.grad_scaler(enabled=False)
    optimizer.zero_grad()
    model(torch.randn(4, 3)).sum().backward()
    optimizer.step()
    states = [m.training for m in model.modules()]
    gradients = [p.requires_grad for p in model.parameters()]
    parameters = [id(p) for p in model.parameters()]
    weights = copy.deepcopy(model.state_dict())
    opt = copy.deepcopy(optimizer.state_dict())
    scale = copy.deepcopy(scaler.state_dict())
    with pytest.raises(RuntimeError, match="evaluation failed"):
        with evaluation_context(model):
            assert platform.get_platform() is context and context.purpose == "training"
            assert not torch.is_grad_enabled()
            assert all(not m.training for m in model.modules())
            model(torch.randn(4, 3))
            raise RuntimeError("evaluation failed")
    assert states == [m.training for m in model.modules()]
    assert gradients == [p.requires_grad for p in model.parameters()]
    assert parameters == [id(p) for p in model.parameters()]
    for name, value in weights.items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    assert opt["param_groups"] == optimizer.state_dict()["param_groups"]
    for param, values in opt["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, optimizer.state_dict()["state"][param][name], rtol=0, atol=0)
    assert scale == scaler.state_dict()


def test_nvidia_binary_cache_cannot_escape_vendor_guard(monkeypatch):
    from up_platform.backends.nvidia import native
    monkeypatch.setattr(native, "_load_binary", lambda: "cached-binary")
    configure_platform({"backend": "nvidia"}, purpose="training", reset=True)
    assert native.load_extension() == "cached-binary"
    configure_platform({"backend": "metax"}, purpose="training", reset=True)
    with pytest.raises(CapabilityError, match="not enabled for metax"):
        native.load_extension()


def test_roi_alignment_convention_is_saved_for_backward(monkeypatch):
    from up_platform.backends.nvidia.ops import roi_align
    calls = []
    def forward(aligned, h, w, scale, sample, features, rois, output):
        calls.append(("forward", aligned))
        output.fill_(1.)
    def backward(aligned, h, w, scale, sample, grad_output, rois, grad_input):
        calls.append(("backward", aligned))
        grad_input.fill_(2.)
    monkeypatch.setattr(roi_align, "roi_align", SimpleNamespace(
        forward_avg_cuda=forward, forward_cpu=forward, backward_avg_cuda=backward))
    monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
    ctx = configure_platform({"backend": "nvidia"}, purpose="training", reset=True)
    value = torch.ones(1, 1, 4, 4, requires_grad=True)
    conventions = {"aligned": True}
    result = ctx.call("roi_align", value, torch.zeros(1, 5), 2, 2, 1., 2, "AVG", **conventions)
    conventions["aligned"] = False
    result.sum().backward()
    assert calls == [("forward", True), ("backward", True)]
    assert torch.equal(value.grad, torch.full_like(value, 2.))
    # Mock kernels test argument/state routing only, not RoI numerical correctness.


def test_static_platform_boundaries():
    # The installed package alone must contain no reverse UP imports.
    import ast
    root = Path(platform.__file__).parent
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "up" and not (node.module or "").startswith("up.")
            if isinstance(node, ast.Import):
                assert all(name.name != "up" and not name.name.startswith("up.") for name in node.names)
