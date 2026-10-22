"""Legacy SDK contracts use a test transport, never a fabricated hardware pass."""
import torch

from up_platform import configure_platform


def test_sync_legacy_transport_keeps_model_and_parameter_identity(monkeypatch):
    from up_platform.training import legacy
    class Transport:
        def __init__(self):
            self.broadcasts = []
            self.reductions = []
            self.waits = 0
        def broadcast(self, value, src):
            self.broadcasts.append(src)
        def allreduce(self, value):
            self.reductions.append(value.clone())
        def synchronize(self):
            self.waits += 1
    transport = Transport()
    monkeypatch.setattr(legacy, "link", transport)
    ctx = configure_platform({"backend": "cpu"}, purpose="training", reset=True)
    model = torch.nn.Linear(2, 2)
    identities = [id(p) for p in model.parameters()]
    assert ctx.training.wrap_legacy_model(model, master_rank=3) is model
    assert identities == [id(p) for p in model.parameters()]
    assert transport.broadcasts == [3, 3]
    model(torch.ones(1, 2)).sum().backward()
    ctx.training.reduce_gradients(model)
    assert len(transport.reductions) == 2
    ctx.training.reduce_gradients(model, sync=False)
    assert transport.waits == 1


def test_async_legacy_hooks_are_retained_until_backward(monkeypatch):
    from up_platform.training import legacy
    calls = []
    class Transport:
        def broadcast(self, value, src):
            pass
        def allreduce_async(self, name, value):
            calls.append((name, value.clone()))
    monkeypatch.setattr(legacy, "link", Transport())
    ctx = configure_platform({"backend": "cpu"}, purpose="training", reset=True)
    model = torch.nn.Linear(2, 2)
    assert ctx.training.wrap_legacy_model(model, sync=False) is model
    assert len(model._grad_accs) == 2
    model(torch.ones(1, 2)).sum().backward()
    assert {name for name, value in calls} == {"weight", "bias"}
