import importlib
import inspect
from pathlib import Path

import pytest
import torch

from up_platform.backends.cpu import build


def test_vendor_bindings_are_explicit():
    root = Path(__file__).resolve().parents[2] / "up_platform" / "ops"
    for source in root.glob("*/base.py"):
        module = importlib.import_module(f"up_platform.ops.{source.parent.name}.base")
        base = next(value for name, value in vars(module).items() if name.endswith("Ops"))
        for vendor in ("nvidia", "metax"):
            impl = importlib.import_module(f"up_platform.ops.{source.parent.name}.{vendor}")
            cls = getattr(impl, vendor.capitalize() + base.__name__)
            assert not inspect.isabstract(cls)
            for name in base.__abstractmethods__:
                assert name in cls.__dict__, (vendor, source, name)


def test_original_module_identity_rng_parameters_and_gradients():
    platform = build({}, "training")
    for category, name, args, shape in [
        ("matrix", "Linear", (4, 3), (2, 4)),
        ("convolution", "Conv2d", (3, 4, 3), (2, 3, 8, 8)),
        ("normalization", "BatchNorm2d", (3,), (2, 3, 8, 8)),
        ("normalization", "LayerNorm", (4,), (2, 4)),
    ]:
        torch.manual_seed(131)
        reference = getattr(torch.nn, name)(*args)
        ref_rng = torch.get_rng_state()
        torch.manual_seed(131)
        candidate = getattr(platform.category(category), name)(*args)
        assert type(candidate) is type(reference)
        assert torch.equal(torch.get_rng_state(), ref_rng)
        assert list(candidate.state_dict()) == list(reference.state_dict())
        for key, value in reference.state_dict().items():
            assert torch.equal(candidate.state_dict()[key], value)
        x = torch.randn(shape, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        a, b = reference(x), candidate(y)
        assert torch.equal(a, b)
        a.sum().backward()
        b.sum().backward()
        assert torch.equal(x.grad, y.grad)
        for left, right in zip(reference.parameters(), candidate.parameters()):
            assert torch.equal(left.grad, right.grad)


def test_validation_restores_heterogeneous_modes_and_does_not_update_state():
    platform = build({}, "training")
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 2))
    model.train()
    model[1].eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    modes = [m.training for m in model.modules()]
    with pytest.raises(ValueError):
        with platform.execution.validation(model):
            assert platform.purpose == "training"
            assert not torch.is_grad_enabled()
            assert not any(m.training for m in model.modules())
            model(torch.randn(4, 3))
            raise ValueError("test restoration on failure")
    assert [m.training for m in model.modules()] == modes
    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value)


def test_missing_capabilities_are_not_verified():
    platform = build({}, "training")
    caps = platform.capabilities()
    assert caps["matrix.linear"].registered
    assert not caps["matrix.linear"].verified
    assert not caps["matrix.linear"].executed
    assert not caps["pointcloud.roiaware_pool3d"].registered
    with pytest.raises(RuntimeError, match="roiaware_pool3d"):
        platform.require(["pointcloud.roiaware_pool3d"])
