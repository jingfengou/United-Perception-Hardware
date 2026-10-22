"""The GEMM fix stays vendor-local, optional, FP32 and outside model code."""
import subprocess
import sys
from types import SimpleNamespace
import copy
import importlib
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("purpose", ["training", "inference"])
def test_v2_lifecycle_cpu_passthrough_and_lazy_kernel(purpose):
    from up_platform import BackendConfigurationError, configure_platform
    from torch.utils._python_dispatch import TorchDispatchMode
    before = "triton" in sys.modules
    try:
        ctx = configure_platform({"backend": "metax", "numerics": "metax_deterministic_v2"},
                                 purpose=purpose, reset=True)
        assert ctx.numerics_state()["profile"] == "metax_deterministic_v2"
        assert ctx.numerics_state()["rng"] == "vendor-native"
        assert "FP32" in ctx.numerics_state()["accumulation"]
        assert ctx is configure_platform({"backend": "metax", "numerics": "metax_deterministic_v2"}, purpose=purpose)
        a, b = torch.arange(6.).reshape(2, 3), torch.ones(3, 2)
        assert torch.equal(a @ b, torch.tensor([[3., 3.], [12., 12.]]))
        assert ctx.numerics_state()["operations"] == {}
        assert ("triton" in sys.modules) == before
        with pytest.raises(BackendConfigurationError, match="Reset"):
            configure_platform({"backend": "metax", "numerics": "metax_deterministic_v1"}, purpose=purpose)

        class Nested(TorchDispatchMode):
            def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
                return operation(*args, **(kwargs or {}))

        with Nested():
            with pytest.raises(BackendConfigurationError, match="nested"):
                configure_platform({"backend": "cpu"}, purpose=purpose, reset=True)
    finally:
        configure_platform({"backend": "cpu"}, purpose=purpose, reset=True)


def test_v2_does_not_enable_nvidia_or_change_default_metax():
    from up_platform import BackendConfigurationError, configure_platform
    try:
        with pytest.raises(BackendConfigurationError, match="Unknown numerical profile"):
            configure_platform({"backend": "nvidia", "numerics": "metax_deterministic_v2"},
                               purpose="training", reset=True)
        ctx = configure_platform({"backend": "metax"}, purpose="training", reset=True)
        assert ctx.numerics_state()["profile"] == "native"
    finally:
        configure_platform({"backend": "cpu"}, purpose="training", reset=True)


def test_operand_contract_rejects_before_loading_triton():
    from up_platform.backends.metax.fixed_gemm import validate_operands
    from up_platform.errors import CapabilityError
    a = SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.float32, layout=torch.strided,
                        ndim=2, shape=(4, 600))
    b = SimpleNamespace(device=a.device, dtype=a.dtype, layout=a.layout, ndim=2, shape=(600, 256))
    validate_operands(a, b)
    for changes in ({"device": torch.device("cpu")}, {"device": torch.device("cuda:1")},
                    {"dtype": torch.float16}, {"layout": torch.sparse_coo}, {"ndim": 3}):
        with pytest.raises(CapabilityError, match="2D GPU FP32"):
            validate_operands(a, SimpleNamespace(**{**vars(b), **changes}))
    with pytest.raises(ValueError, match="matrix shapes"):
        validate_operands(a, SimpleNamespace(**{**vars(b), "shape": (599, 256)}))


def test_device_initialization_does_not_load_kernel_or_other_purpose():
    code = """
import sys
from up_platform import configure_platform
ctx = configure_platform({'backend': 'metax', 'numerics': 'metax_deterministic_v2'}, purpose='inference')
assert not any(n == 'triton' or n.endswith('fixed_gemm_kernel') for n in sys.modules)
assert not any(n.startswith('up_platform.training') for n in sys.modules)
assert not any(n == 'up' or n.startswith('up.') for n in sys.modules)
configure_platform({'backend': 'cpu'}, purpose='inference', reset=True)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_gpu_out_variant_is_rejected_not_silently_native():
    from up_platform.backends.metax.determinism_v2 import DeterministicGemm
    from up_platform.errors import CapabilityError
    a = SimpleNamespace(device=torch.device("cuda:0"))
    with pytest.raises(CapabilityError, match="not aten.mm.out"):
        DeterministicGemm().__torch_dispatch__(torch.ops.aten.mm.out, (), (a, a), {"out": a})


def test_complete_report_comparison_rejects_protocol_drift_and_detects_gradients(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmark/hardware"))
    collector = importlib.import_module("collect_metax_gemm")
    row = {"loss": 1., "input_sha256": "input", "raw_gradient_sha256": "raw",
           "clipped_gradient_sha256": "clipped", "weight_sha256": "weights"}
    reference = {"completed": True, "steps": [dict(row) for _ in range(720)],
                 "profile": "metax_deterministic_v2", "rtol": 1e-4, "atol": 1e-5,
                 "initial_model_sha256": "initial", "torch": "test", "gpu": "C500"}
    assert collector.compare_reports(reference, copy.deepcopy(reference))["passed"]
    for key, value in (("rtol", 1e-3), ("profile", "native"), ("completed", False)):
        wrong = {**reference, key: value}
        with pytest.raises(AssertionError):
            collector.compare_reports(reference, wrong)
    other = copy.deepcopy(reference)
    other["steps"][57]["raw_gradient_sha256"] = "different"
    result = collector.compare_reports(reference, other)
    assert not result["passed"] and not result["exact"]["raw_gradient_sha256"]
    other["steps"][57]["loss"] = float("nan")
    with pytest.raises(AssertionError):
        collector.compare_reports(reference, other)
