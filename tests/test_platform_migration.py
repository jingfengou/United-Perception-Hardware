"""UP facade/integration checks; no vendor SDK, GPU or full application imports."""
import ast
import copy
import importlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch

from up.accelerator import configure_accelerator, get_accelerator
from up_platform import configure_platform, register_op
from up_platform.registry import OPERATIONS

ROOT = Path(__file__).resolve().parents[1]


def test_common_cannot_import_experimental_policies():
    tool = runpy.run_path(str(ROOT / "tools/platform/check_hardware_boundaries.py"))
    path = "packages/up-platform/src/up_platform/common/runtime.py"
    for source in ("from up_platform.experimental.numerics.mode import PortableNumerics",
                   "import importlib\nimportlib.import_module('up_platform.experimental.numerics.mode')"):
        findings = tool["check_platform_source"](source, path)
        assert any(row["rule"] == "experimental_dependency" for row in findings)


@pytest.fixture(autouse=True)
def isolate():
    old = copy.deepcopy(OPERATIONS)
    configure_platform({"backend": "cpu"}, purpose="training", reset=True)
    yield
    OPERATIONS.clear()
    OPERATIONS.update(old)
    configure_platform({"backend": "cpu"}, purpose="training", reset=True)


def runner_method(name, **namespace):
    tree = ast.parse((ROOT / "up/runner/base_runner.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseRunner")
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "up/runner/base_runner.py", "exec"), namespace)
    return namespace[name]


def test_actual_runner_eval_restores_mixed_module_states():
    from up_platform.training.evaluation import preserve_evaluation_state
    method = runner_method("evaluate", torch=torch, preserve_evaluation_state=preserve_evaluation_state,
                           env=SimpleNamespace(is_master=lambda: True), logger=SimpleNamespace(info=lambda x: None),
                           json=json, barrier=lambda: None)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.4)).train()
    model[1].eval()
    before = [value.training for value in model.modules()]
    class Runner:
        evaluate = method
        def __init__(self):
            self.model, self.ema = model, None
            self.data_loaders = {"test": SimpleNamespace(dataset=SimpleNamespace(evaluate=lambda *args: {"ok": 1}))}
        def inference(self):
            self.model.eval()
            assert not torch.is_grad_enabled()
            return "unused", []
        def _hooks(self, *args):
            pass
        def set_cur_eval_iter(self):
            pass
    runner = Runner()
    context = get_accelerator().context
    assert runner.evaluate() == {"ok": 1}
    assert [value.training for value in model.modules()] == before
    assert get_accelerator().context is context and context.purpose == "training"


def test_actual_runner_inference_does_not_build_scaler_or_ddp():
    def forbidden(*args, **kwargs):
        raise AssertionError("training-only object created for inference")
    configure_accelerator(backend="cpu", purpose="inference", reset=True)
    flag = SimpleNamespace(fp16=False)
    method = runner_method("prepare_fp16", get_accelerator=get_accelerator, FP16_FLAG=flag)
    runner = SimpleNamespace(config={"runtime": {"fp16": True}}, backend="dist", fp16=False, training=False)
    method(runner)
    assert runner.fp16 and flag.fp16 and not hasattr(runner, "scaler")
    method = runner_method("prepare_dist_model", env=SimpleNamespace(distributed=True),
                           get_accelerator=get_accelerator, DDP=forbidden, DistModule=forbidden)
    method(runner)


def test_old_callable_and_lazy_factory_registration_remain_supported(tmp_path, monkeypatch):
    from up.accelerator import register_backend, TorchAccelerator
    from up_platform.registry import BACKENDS
    names = ("legacy_callable_factory", "legacy_lazy_factory")
    try:
        register_backend(names[0], lambda device=None: TorchAccelerator(device))
        assert configure_accelerator(backend=names[0], reset=True).device_type == "cpu"
        (tmp_path / "lazy_legacy_factory.py").write_text(
            'from up.accelerator import TorchAccelerator\n'
            'def create(device=None):\n    return TorchAccelerator(device)\n', encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        register_backend(names[1], "lazy_legacy_factory:create")
        assert "lazy_legacy_factory" not in __import__("sys").modules
        assert configure_accelerator(backend=names[1], reset=True).name == names[1]
    finally:
        for name in names:
            BACKENDS.pop(name, None)


def test_legacy_training_only_override_is_not_shadowed():
    from up.accelerator import register_op as old_register
    old_register("nvidia", "sigmoid_focal_loss", lambda x: x * 2, backward=True, replace=True)
    ctx = configure_accelerator(backend="nvidia", reset=True)
    assert ctx.call("sigmoid_focal_loss", torch.tensor(2.)).item() == 4.


@pytest.mark.parametrize("command,purpose", [("train", "training"), ("inference", "inference"), ("to_onnx", "inference")])
def test_cli_main_and_parent_fix_purpose(command, purpose):
    tree = ast.parse((ROOT / "up/commands" / (command + ".py")).read_text(encoding="utf-8"))
    for function, callee in (("main", "activate_args"), ("_main", "prepare_args")):
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
        call = next(node for node in ast.walk(node) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == callee)
        assert next(keyword.value.value for keyword in call.keywords if keyword.arg == "purpose") == purpose


def test_old_apply_routes_same_forward_and_gradients_and_coordinate_flags():
    from up.extensions.python.roi_align import RoIAlignFunction
    from up.flags import ALIGNED_FLAG
    flags = ALIGNED_FLAG.aligned, ALIGNED_FLAG.offset
    def operation(features, rois, h, w, scale, sample, mode, aligned=False):
        return features * (2 if aligned else 3)
    register_op("cpu", "roi_align", operation, scope="common", backward=True, conventions=("aligned",))
    ctx = configure_platform({"backend": "cpu"}, purpose="training", reset=True)
    x = torch.ones(1, 1, 2, 2, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    try:
        ALIGNED_FLAG.aligned = True
        a = RoIAlignFunction.apply(x, torch.zeros(1, 5), 2, 2, 1., 0, "AVG")
        b = ctx.ops.roi_align(y, torch.zeros(1, 5), 2, 2, 1., 0, "AVG", aligned=True)
        a.sum().backward()
        b.sum().backward()
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    finally:
        ALIGNED_FLAG.aligned, ALIGNED_FLAG.offset = flags


def test_legacy_and_moved_symbolic_nodes_match():
    class Graph:
        def op(self, *args, **kwargs):
            return args, kwargs
    cases = [
        ("roi_align", "RoIAlignFunction", ("features", "rois", 2, 3, .25, 2, "AVG")),
        ("psroi_align", "PSRoIAlignFunction", ("features", "rois", 2, .25, 3, 2)),
        ("psroi_pool", "PSRoIPoolFunction", ("features", "rois", 2, .25, 3)),
    ]
    for module, name, arguments in cases:
        old = getattr(importlib.import_module("up.extensions.python." + module), name)
        new = getattr(importlib.import_module("up_platform.backends.nvidia.ops." + module), name)
        assert old.symbolic(Graph(), *arguments) == new.symbolic(Graph(), *arguments)


def test_complete_platform_index_has_semantics_and_no_drift():
    tool = runpy.run_path(str(ROOT / "tools/platform/export_platform_contract.py"))
    for path, expected in tool["outputs"]().items():
        assert path.read_text(encoding="utf-8") == expected
    from up_platform.contracts import build_contract
    contract = build_contract()
    assert len(contract["runtime"]) == 31 and len(contract["operations"]) == 73
    for item in contract["operations"]:
        assert all(item[key] for key in ("inputs", "returns", "device_and_layout", "empty_input", "state_updates", "gradients", "export"))


@pytest.mark.parametrize("source,path,rule", [
    ("import up", "packages/up-platform/src/up_platform/common/bad.py", "platform_reverse_dependency"),
    ("from up import flags", "packages/up-platform/src/up_platform/shared/bad.py", "platform_reverse_dependency"),
    ("from up_platform.inference.backend import InferenceBackend", "packages/up-platform/src/up_platform/training/bad.py", "cross_purpose_import"),
    ("from up_platform.training.backend import TrainingBackend", "packages/up-platform/src/up_platform/backends/new/inference.py", "cross_purpose_import"),
    ("from ...inference.backend import InferenceBackend", "packages/up-platform/src/up_platform/training/nested/bad.py", "cross_purpose_import"),
])
def test_checker_rejects_reverse_and_cross_purpose_imports(source, path, rule):
    tool = runpy.run_path(str(ROOT / "tools/platform/check_hardware_boundaries.py"))
    assert rule in [item["rule"] for item in tool["check_platform_source"](source, path)]
