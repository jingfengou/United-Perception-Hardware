import sys

import pytest
import torch

from up_platform import factory
from up_platform.base import PlatformBackend
from up_platform.backends.torch_device import TorchDevice
from up_platform.ops.matrix.cpu import CpuMatrixOps
from up_platform.training.cpu import CpuTraining
from up_platform.inference.cpu import CpuInference


@pytest.fixture(autouse=True)
def isolated_selection(monkeypatch):
    monkeypatch.setattr(factory, '_context', None)
    monkeypatch.setattr(factory, '_selection', None)
    monkeypatch.setattr(factory, '_FACTORIES', dict(factory._FACTORIES))


def test_new_device_single_override_and_fixed_purpose():
    calls = []
    class DemoDevice(TorchDevice):
        vendor = 'demo'
        device_type = 'privateuseone'
    class DemoMatrix(CpuMatrixOps):
        @staticmethod
        def linear(input, weight, bias=None):
            calls.append('demo.linear')
            return torch.nn.functional.linear(input, weight, bias)
    def build(config, purpose):
        device = DemoDevice()
        execution = CpuTraining(device) if purpose == 'training' else CpuInference(device)
        return PlatformBackend(device, purpose, {'matrix': DemoMatrix()}, execution)
    factory.register_backend('demo', build)
    p = factory.configure_platform({'backend': 'demo'}, purpose='training')
    assert p.device.device().type == 'privateuseone'
    assert p.category('matrix').mm is CpuMatrixOps.mm
    x = torch.randn(2, 3, requires_grad=True)
    w = torch.randn(4, 3, requires_grad=True)
    p.category('matrix').linear(x, w).sum().backward()
    assert calls == ['demo.linear'] and x.grad is not None and w.grad is not None
    assert factory.configure_platform({'backend': 'demo'}, purpose='training') is p
    with pytest.raises(RuntimeError, match='fixed'):
        factory.configure_platform({'backend': 'demo'}, purpose='inference')
    with pytest.raises(RuntimeError, match='before'):
        factory.register_backend('late', build)
    with pytest.raises(RuntimeError, match='not registered'):
        p.require(['detection.roi_align'])
    with pytest.raises(RuntimeError):
        p.category('matrix').linear(x, torch.ones(4, 3, dtype=torch.int64))


def test_training_and_inference_modules_load_separately():
    import subprocess
    for purpose, excluded in [('training', 'inference'), ('inference', 'training')]:
        source = f"from up_platform import configure_platform; import sys; p=configure_platform({{'backend':'cpu'}},purpose='{purpose}'); assert not any(n.startswith('up_platform.{excluded}.') for n in sys.modules); assert 'up' not in sys.modules"
        subprocess.run([sys.executable, '-c', source], check=True)


def test_inference_has_no_optimizer_or_validation_switch():
    p = factory.configure_platform({'backend': 'cpu'}, purpose='inference')
    assert not hasattr(p.execution, 'optimizer')
    assert not hasattr(p.execution, 'validation')
    with pytest.raises(ValueError, match='native'):
        factory.configure_platform({'backend': 'cpu', 'numerics': 'portable_v1'}, purpose='inference')


def test_dependency_availability_does_not_claim_verification(monkeypatch):
    p = factory.configure_platform({'backend': 'cpu'}, purpose='training')
    import up_platform.dependencies
    monkeypatch.setattr(up_platform.dependencies, 'available', lambda _: False)
    caps = p.capabilities()
    assert caps['matrix.linear'].registered and not caps['matrix.linear'].dependency_available
    assert not caps['matrix.linear'].verified
    with pytest.raises(RuntimeError, match='dependency'):
        p.require(['matrix.linear'])
