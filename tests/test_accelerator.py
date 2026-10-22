import importlib
import json
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import pytest
import torch

from up.accelerator import (
    BackendConfigurationError, CapabilityError, TorchAccelerator,
    capabilities, configure_accelerator, get_accelerator, ops,
    register_backend, register_op,
)
from up.accelerator.registry import BACKENDS, BACKEND_PROBES, OPERATIONS, operation

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("backend", ("nvidia", "metax"))
def test_cuda_shaped_convolution_controls_are_shared_and_restore(backend):
    from up.accelerator.backends.torch_backends import MetaXAccelerator, NvidiaAccelerator
    provider = (NvidiaAccelerator if backend == "nvidia" else MetaXAccelerator)(device="cuda")
    original = torch.backends.cudnn.enabled
    try:
        provider.configure_convolution(False)
        assert torch.backends.cudnn.enabled is False
        with pytest.raises(RuntimeError, match="restore probe"):
            with provider.convolution_flags(True):
                assert torch.backends.cudnn.enabled is True
                raise RuntimeError("restore probe")
        assert torch.backends.cudnn.enabled is False
        provider.configure_convolution(True)
        assert torch.backends.cudnn.enabled is True
    finally:
        torch.backends.cudnn.enabled = original


@pytest.fixture(autouse=True)
def restore_registry():
    backends, operations, probes = dict(BACKENDS), dict(OPERATIONS), dict(BACKEND_PROBES)
    configure_accelerator(backend="cpu", reset=True)
    yield
    BACKENDS.clear()
    BACKENDS.update(backends)
    OPERATIONS.clear()
    OPERATIONS.update(operations)
    BACKEND_PROBES.clear()
    BACKEND_PROBES.update(probes)
    configure_accelerator(backend="cpu", reset=True)


def test_core_import_does_not_load_application_or_vendor_sdks():
    script = """
import sys
import up.accelerator
assert not any(name.startswith(('torch_npu', 'torch_mlu', 'spconv', 'spring',
                              'mqbench', 'msbench', 'torchvision', 'numba.cuda', 'up.tasks',
                              'up.extensions._C')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)


def test_recursive_transfer_preserves_structure():
    Pair = namedtuple("Pair", "first second")
    data = {"image": torch.randn(2, 3), "nested": Pair(torch.ones(2), [torch.zeros(1), "metadata"])}
    result = get_accelerator().to_device(data)
    assert isinstance(result["nested"], Pair)
    assert result["nested"].second[1] == "metadata"
    assert result["image"].device.type == "cpu"


@pytest.mark.parametrize("offset,expected", [(0, [0, 2]), (1, [0, 2])])
def test_portable_nms_contract(offset, expected):
    from up.flags import ALIGNED_FLAG
    original = ALIGNED_FLAG.offset
    ALIGNED_FLAG.offset = offset
    try:
        boxes = torch.tensor([[0., 0., 10., 10., .9], [1., 1., 9., 9., .8], [20., 20., 25., 25., .7]])
        assert ops.naive_nms(boxes, .5).tolist() == expected
        assert ops.naive_nms(boxes[:0], .5).shape == (0,)
    finally:
        ALIGNED_FLAG.offset = original


@pytest.mark.parametrize("offset", [0, 1])
def test_torchvision_nms_keeps_score_order_and_exclusive_coordinates(offset):
    from up.flags import ALIGNED_FLAG
    original = ALIGNED_FLAG.offset
    ALIGNED_FLAG.offset = offset
    try:
        boxes = torch.tensor([[0., 0., 1., 1.], [1., 1., 2., 2.], [4., 4., 5., 5.]])
        scores = torch.tensor([.8, .9, .7])
        assert ops.torchvision_nms(boxes, scores, .1).tolist() == [1, 0, 2]
        assert ops.torchvision_nms(boxes[:0], scores[:0], .1).dtype == torch.int64
    finally:
        ALIGNED_FLAG.offset = original


@pytest.mark.parametrize("mode", ["IoU", "IoF", "IoS"])
def test_portable_overlap_contract(mode):
    box = torch.tensor([[0., 0., 3., 3.], [10., 10., 12., 12.]])
    torch.testing.assert_close(ops.gpu_iou_overlap(box, box, mode), torch.eye(2))
    assert ops.gpu_iou_overlap(box[:0], box).shape == (0,)


def test_shared_implementation_and_single_operation_override():
    class FutureBackend(TorchAccelerator):
        name = "future_test"
    register_backend(FutureBackend.name, FutureBackend)
    register_op("portable", "test_scale", lambda x: x * 2, backward=True)
    register_op("future_test", "test_scale", lambda x: x * 3, backward=True)
    configure_accelerator(backend="future_test", reset=True)
    value = torch.tensor(2., requires_grad=True)
    class UnchangedModel(torch.nn.Module):
        def forward(self, x):
            return ops.call("test_scale", x)
    result = UnchangedModel()(value)
    result.backward()
    assert result.detach().item() == 6 and value.grad.item() == 3
    assert operation(get_accelerator(), "naive_nms") is operation(TorchAccelerator(), "naive_nms")


def test_different_device_name_routes_without_device_type_conditionals():
    class FutureDevice(TorchAccelerator):
        name = "private_device_test"
        device_type = "privateuseone"
    register_backend(FutureDevice.name, FutureDevice)
    register_op(FutureDevice.name, "routing_probe", lambda: "private device provider")
    accelerator = configure_accelerator(backend=FutureDevice.name, device="privateuseone:1", reset=True)
    assert str(accelerator.device) == "privateuseone:1"
    class UnchangedModel(torch.nn.Module):
        def forward(self):
            return ops.call("routing_probe")
    assert UnchangedModel()() == "private device provider"
    # This is a dispatch contract test; it does not simulate real device execution.


def test_same_device_name_does_not_enable_nvidia_kernels():
    accelerator = configure_accelerator(backend="metax", reset=True)
    assert accelerator.device_type == "cuda"
    assert capabilities()["operations"]["naive_nms"]["owner"] == "portable"
    with pytest.raises(CapabilityError, match="metax.*roi_align"):
        ops.roi_align(None)


def test_backend_switch_requires_explicit_reset():
    with pytest.raises(BackendConfigurationError, match="already configured"):
        configure_accelerator(backend="metax")


def test_unknown_backend_is_actionable():
    with pytest.raises(BackendConfigurationError, match="Unknown accelerator"):
        configure_accelerator(backend="unknown", reset=True)


def test_probe_ambiguity_is_not_resolved_by_device_name():
    register_backend("first_probe", TorchAccelerator, probe=lambda: True)
    register_backend("second_probe", TorchAccelerator, probe=lambda: True)
    with pytest.raises(BackendConfigurationError, match="Ambiguous"):
        configure_accelerator(backend="auto", reset=True)


def test_environment_version_mismatch_is_actionable():
    with pytest.raises(BackendConfigurationError, match="requires python"):
        configure_accelerator({"runtime": {"accelerator": {
            "backend": "cpu", "requirements": {"python": "<2"}}}}, reset=True)


def test_foreign_device_is_rejected():
    with pytest.raises(BackendConfigurationError, match="requires"):
        configure_accelerator(backend="cpu", device="cuda:0", reset=True)


def test_device_resolution_preserves_torch_device_and_integer_contract():
    accelerator = get_accelerator()
    assert accelerator.resolve_device(2) == torch.device("cpu:2")
    assert accelerator.resolve_device(torch.device("cpu")) == torch.device("cpu")
    assert accelerator.resolve_device("cpu:1") == torch.device("cpu:1")
    with pytest.raises(BackendConfigurationError, match="foreign device"):
        accelerator.resolve_device(torch.device("cuda:0"))


def test_alias_can_reuse_runtime_class_and_override_one_operation():
    register_backend("runtime_alias", TorchAccelerator)
    register_op("runtime_alias", "alias_probe", lambda: "overridden")
    configure_accelerator(backend="runtime_alias", reset=True)
    assert get_accelerator().name == "runtime_alias"
    assert ops.call("alias_probe") == "overridden"


def test_documented_example_plugin():
    configure_accelerator({"runtime": {"device": "cpu", "accelerator": {
        "backend": "example_cpu", "plugins": ["up.accelerator.example_plugin"]}}}, reset=True)
    assert capabilities()["operations"]["naive_nms"]["owner"] == "example_cpu"
    assert capabilities()["operations"]["gpu_iou_overlap"]["owner"] == "portable"
    with pytest.raises(CapabilityError, match="does not support"):
        ops.naive_nms(torch.zeros(1, 5, dtype=torch.float64), .5)


def test_autocast_rejects_invalid_precision():
    with pytest.raises(CapabilityError, match="autocast does not support"):
        get_accelerator().autocast(dtype=torch.float64)


def test_duplicate_registration_is_rejected():
    with pytest.raises(ValueError):
        register_backend("cpu", TorchAccelerator)
    with pytest.raises(ValueError):
        register_op("portable", "naive_nms", lambda: None)


def test_missing_capability_never_returns_placeholder():
    with pytest.raises(CapabilityError, match="cpu.*roi_align"):
        ops.roi_align(torch.zeros(1), torch.zeros(1))


def test_rng_round_trip():
    accelerator = get_accelerator()
    accelerator.manual_seed(123)
    expected = torch.rand(3)
    accelerator.manual_seed(123)
    torch.testing.assert_close(torch.rand(3), expected)
    state = accelerator.get_rng_state()
    first = torch.rand(3)
    accelerator.set_rng_state(state)
    torch.testing.assert_close(first, torch.rand(3))


def test_cpu_runtime_and_disabled_autocast():
    accelerator = get_accelerator()
    assert accelerator.is_available() and accelerator.device_count() == 1
    assert accelerator.memory_allocated() == 0
    accelerator.synchronize()
    accelerator.empty_cache()
    with accelerator.autocast(enabled=False):
        assert torch.ones(1).dtype == torch.float32


def test_unavailable_device_does_not_silently_disable_amp():
    class MissingDevice(TorchAccelerator):
        def is_available(self):
            return False
    register_backend("missing_device", MissingDevice)
    accelerator = configure_accelerator(backend="missing_device", reset=True)
    with pytest.raises(CapabilityError, match="available device"):
        accelerator.autocast()
    with pytest.raises(CapabilityError, match="available device"):
        accelerator.grad_scaler()


def test_distributed_default_and_explicit_transport_keep_distinct_meaning():
    from up.accelerator import distributed
    class TransportBackend(TorchAccelerator):
        communication_backend = "test_collective"
    register_backend("transport_contract", TransportBackend)
    register_op("transport_contract", "distributed.init_process_group", lambda backend: backend)
    register_op("transport_contract", "distributed.cpu_group", lambda provider: ("test_group", provider))
    configure_accelerator(backend="transport_contract", reset=True)
    assert distributed.init_process_group() == "test_collective"
    assert distributed.init_process_group(backend="nccl") == "nccl"
    assert distributed.cpu_group() == ("test_group", "dist")


def test_no_native_extension_loaded_by_capability_query():
    configure_accelerator(backend="nvidia", reset=True)
    result = capabilities()
    assert result["operations"]["roi_align"]["backward"] is True
    assert result["operations"]["roipoint_pool3d"]["backward"] is False
    assert "up.extensions._C" not in sys.modules


def test_missing_optional_provider_fails_explicitly():
    with pytest.raises(CapabilityError, match="no provider"):
        get_accelerator().dependency("sparse")


def test_deferred_layer_preserves_native_mro_and_state_dict():
    from types import SimpleNamespace
    from up.accelerator.layers import LinkSyncBatchNorm
    class NativeBN(torch.nn.BatchNorm2d):
        def __init__(self, *args, **kwargs):
            kwargs.pop("var_mode", None)
            super().__init__(*args, **kwargs)
    provider = SimpleNamespace(nn=SimpleNamespace(SyncBatchNorm2d=NativeBN),
                               syncbnVarMode_t=SimpleNamespace(L2=2))
    class TestBackend(TorchAccelerator):
        name = "deferred_layer_test"
        def dependency(self, name):
            return provider
    register_backend(TestBackend.name, TestBackend)
    configure_accelerator(backend=TestBackend.name, reset=True)
    class DerivedBN(LinkSyncBatchNorm):
        def __init__(self):
            super().__init__(3, var_mode=None)
    layer = DerivedBN()
    assert isinstance(layer, NativeBN) and isinstance(layer, DerivedBN)
    assert set(layer.state_dict()) == set(NativeBN(3).state_dict())
    assert layer(torch.randn(2, 3, 4, 4)).shape == (2, 3, 4, 4)


def test_sync_layer_can_override_forward_without_parameter_nesting():
    from up.accelerator.layers import SyncBatchNorm
    class TestBackend(TorchAccelerator):
        name = "layer_override_test"
    register_backend(TestBackend.name, TestBackend)
    register_op(TestBackend.name, "sync_batch_norm_forward", lambda module, x: x * module.weight[0])
    configure_accelerator(backend=TestBackend.name, reset=True)
    layer = SyncBatchNorm(3)
    assert set(layer.state_dict()) == set(torch.nn.SyncBatchNorm(3).state_dict())
    value = torch.randn(2, 3, 4, 4, requires_grad=True)
    layer(value).sum().backward()
    assert value.grad is not None and layer.weight.grad is not None


def test_sparse_provider_keeps_native_parameter_names():
    from types import SimpleNamespace
    from up.accelerator import sparse

    class SparseValue:
        def __init__(self, features):
            self.features = features

    class NativeConv(sparse.SparseModule):
        def __init__(self, in_channels, out_channels, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(in_channels, out_channels))

        def forward(self, value):
            return SparseValue(value.features @ self.weight)

    provider = SimpleNamespace(SparseModule=sparse.SparseModule, SubMConv3d=NativeConv,
                               conv=SimpleNamespace(SparseConvolution=NativeConv))

    class SparseBackend(TorchAccelerator):
        def dependency(self, name):
            assert name == "sparse"
            return provider

    register_backend("sparse_contract", SparseBackend)
    configure_accelerator(backend="sparse_contract", reset=True)
    layer = sparse.SparseSequential(sparse.SubMConv3d(2, 3, 1), torch.nn.ReLU())
    assert set(layer.state_dict()) == {"0.weight"}
    assert sparse.is_sparse_convolution(layer[0])
    value = torch.ones(4, 2, requires_grad=True)
    result = sparse.features(layer(SparseValue(value)))
    result.sum().backward()
    assert value.grad is not None and layer[0].weight.grad is not None


def test_sparse_tensor_operations_accept_foreign_representation():
    from up.accelerator import sparse
    register_backend("foreign_sparse", TorchAccelerator)
    register_op("foreign_sparse", "sparse.features", lambda value: value["payload"])
    register_op("foreign_sparse", "sparse.replace_feature", lambda value, features: {"payload": features})
    register_op("foreign_sparse", "sparse.to_dense", lambda value: value["payload"].unsqueeze(0))
    configure_accelerator(backend="foreign_sparse", reset=True)
    value = {"payload": torch.ones(2, 3, requires_grad=True)}
    changed = sparse.replace_feature(value, sparse.features(value) * 2)
    sparse.to_dense(changed).sum().backward()
    torch.testing.assert_close(value["payload"].grad, torch.full((2, 3), 2.))


def test_build_without_torch_or_cuda():
    env = dict(__import__("os").environ, UP_BUILD_BACKEND="none")
    result = subprocess.run([sys.executable, "setup.py", "--name"], cwd=ROOT, env=env,
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "UP"


def test_static_hardware_boundary():
    spec = importlib.util.spec_from_file_location("boundary_check", ROOT / "tools/platform/check_hardware_boundaries.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.check_tree() == []


@pytest.mark.parametrize("source", [
    "import torch as t\nt.cuda.empty_cache()",
    "from torch import cuda as runtime\nruntime.synchronize()",
    "x.cuda()", "x.is_cuda", "import spconv.pytorch as sparse",
    "from numba import cuda", "import torch\ntorch.zeros(2, device='npu')",
    "import torch.distributed as d\nd.all_reduce(x)",
    "device = 'cuda'\nx.to(device)",
    "import importlib\nimportlib.import_module('torch_npu')",
    "from up.accelerator.backends.nvidia_ops import nms",
    "from up.extensions import ext",
    "import torch\ntorch.amp.autocast('cuda')",
    "import torch_npu\nx.is_npu",
    "from torchvision.ops import nms",
    "import torchvision as vision\nvision.ops.nms(boxes, scores, .5)",
    "from torchvision.ops import roi_align",
])
def test_static_checker_detects_bypasses(source):
    spec = importlib.util.spec_from_file_location("boundary_check", ROOT / "tools/platform/check_hardware_boundaries.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.check_source(source, "up/test_module.py")
