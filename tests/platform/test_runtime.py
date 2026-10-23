from collections import namedtuple

import pytest
import torch

from up_platform.backends.cpu import CpuDevice
from up_platform.backends.torch_device import TorchDevice
from up_platform.training.cpu import CpuTraining


def test_explicit_device_zero_rng_is_not_replaced_by_current_device():
    calls = []
    class Api:
        def get_rng_state(self, device):
            calls.append(device)
            return torch.tensor([1], dtype=torch.uint8)
        def set_rng_state(self, state, device):
            calls.append(device)
    class Device(TorchDevice):
        device_type = 'privateuseone'
        vendor = 'test'
        api = Api()
    device = Device('privateuseone:1')
    state = device.get_rng_state(0)
    device.set_rng_state(state, 0)
    assert calls == [0, 0]


def test_move_preserves_nested_containers_and_parameter_identity():
    device = CpuDevice()
    Pair = namedtuple('Pair', ['x', 'y'])
    module = torch.nn.Linear(3, 4)
    params = list(module.parameters())
    result = device.move({'pair': Pair(torch.ones(2), [module])})
    assert isinstance(result['pair'], Pair)
    assert result['pair'].y[0] is module
    assert all(a is b for a, b in zip(params, module.parameters()))


def test_validation_restores_mixed_modes_and_does_not_change_state_on_error():
    training = CpuTraining(CpuDevice())
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
    model.train()
    model[1].eval()
    modes = [m.training for m in model.modules()]
    state = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = training.optimizer('Adam', params=model.parameters(), lr=.001)
    with pytest.raises(ValueError, match='test'):
        with training.validation(model):
            assert not torch.is_grad_enabled()
            assert not any(m.training for m in model.modules())
            model(torch.ones(2, 3))
            raise ValueError('test')
    assert [m.training for m in model.modules()] == modes
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    assert optimizer.state_dict()['state'] == {}


def test_legacy_export_uses_symbolic_exporter(monkeypatch):
    from up_platform.inference.torch_impl import TorchInference
    calls = []
    def export(model, args, output, *, dynamo=True, **kwargs):
        calls.append(dynamo)
    monkeypatch.setattr(torch.onnx, 'export', export)
    TorchInference(CpuDevice()).export_onnx(torch.nn.Identity(), (torch.ones(1),), 'unused.onnx')
    assert calls == [False]


def test_metax_sparse_dense_adapts_layout_without_changing_values_or_gradients():
    from up_platform.ops.sparse.metax import MetaxSparseOps
    x = torch.randn(2, 3, 4, 5, 6, requires_grad=True)
    class Tensor:
        def dense(self):
            return x
    ops = MetaxSparseOps()
    assert ops.dense(Tensor(), channels_first=False) is x
    result = ops.dense(Tensor())
    assert result.shape == (2, 6, 3, 4, 5) and result.is_contiguous()
    assert torch.equal(result, x.permute(0, 4, 1, 2, 3))
    result.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_pruning_iteration_does_not_change_masks():
    from up_platform.ops.pruning.cpu import CpuPruningOps
    class Scheduler:
        iteration = 0
        mask = torch.ones(3)
    scheduler = Scheduler()
    ops = CpuPruningOps()
    ops.set_iteration(scheduler, 42)
    assert scheduler.iteration == 42 and torch.equal(scheduler.mask, torch.ones(3))
    assert ops.last_prune_step(scheduler) == 0
    scheduler.get_last_prune_step = lambda: 7
    assert ops.last_prune_step(scheduler) == 7


def test_up_cpu_configuration_is_explicit_not_hardware_auto_detection(monkeypatch):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / 'up/utils/env/platform.py'
    spec = importlib.util.spec_from_file_location('up_config_adapter', path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    monkeypatch.setattr(adapter, 'configure_platform', lambda config, *, purpose: (config, purpose))
    assert adapter.configure_from_up({'runtime': {'device': 'cpu'}}, purpose='inference') == (
        {'backend': 'cpu', 'device': 'cpu'}, 'inference')
    config = {'runtime': {'device': 'cuda', 'platform': {'backend': 'metax'}}}
    assert adapter.configure_from_up(config, purpose='training') == (
        {'backend': 'metax', 'device': 'cuda'}, 'training')


def test_import_time_default_defers_selection_and_preserves_shared_module(monkeypatch):
    from up_platform import layers
    from up_platform.backends.cpu import build
    calls = []
    def unconfigured():
        calls.append('unconfigured')
        raise RuntimeError('not configured')
    monkeypatch.setattr(layers, 'get_platform', unconfigured)
    rng = torch.get_rng_state()
    descriptor = layers.defer_default(layers.Dropout, p=0.)
    assert not calls and torch.equal(rng, torch.get_rng_state())
    monkeypatch.setattr(layers, 'get_platform', lambda: build({}, 'training'))
    first = layers.resolve_default(descriptor)
    assert type(first) is torch.nn.Dropout and first.p == 0.
    assert first is layers.resolve_default(descriptor)
    assert not first.state_dict() and torch.equal(rng, torch.get_rng_state())
    first.eval()
    assert not layers.resolve_default(descriptor).training
    assert layers.resolve_default(first) is first
    assert layers.resolve_default(None) is None
