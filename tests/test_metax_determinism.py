"""Diagnostic controls must keep the original numerical acceptance criterion."""
import importlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def tool(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("diagnose_metax_determinism")


def test_fixed_tolerance_and_small_differences_are_not_bitwise(tool):
    assert (tool.RTOL, tool.ATOL) == (1e-4, 1e-5)
    result = tool.difference(torch.ones(2), torch.ones(2) + 1e-6)
    assert not result["exact"] and result["outside_tolerance"] == 0
    assert tool.difference(torch.zeros(2), torch.ones(2))["outside_tolerance"] == 2


@pytest.mark.parametrize("invalid", [float("inf"), float("nan")])
def test_nonfinite_outputs_are_not_accepted(tool, invalid):
    with pytest.raises(AssertionError, match="Nonfinite"):
        tool.difference(torch.ones(2), torch.full((2,), invalid))


def test_empty_and_unrequested_gradient_are_distinct(tool):
    assert tool.difference(None, None)["exact"]
    missing = tool.difference(None, torch.zeros(1))
    assert not missing["exact"] and missing["outside_tolerance"] == 1
    assert tool.difference(torch.empty(0), torch.empty(0))["exact"]


def test_repeat_uses_fixed_operands_and_preserves_first_return(tool):
    calls = []
    value = torch.arange(4.)

    def operation(operand):
        calls.append(operand)
        return operand * 2, None

    result, stats = tool.repeat_operation(operation, (value,), {}, 10)
    assert stats["exact"] and stats["max_abs"] == [0., 0.]
    assert len(calls) == 10 and all(item is value for item in calls)
    assert torch.equal(result[0], value * 2) and result[1] is None


def test_repeat_detects_small_nondeterminism_without_relaxing_tolerance(tool):
    count = 0

    def operation():
        nonlocal count
        count += 1
        return torch.tensor([count * 1e-7])

    _, stats = tool.repeat_operation(operation, (), {}, 4)
    assert not stats["exact"] and stats["max_abs"][0] > 0
    assert all(row[0]["outside_tolerance"] == 0 for row in stats["rows"])


@pytest.mark.parametrize("groups,stride,padding,dilation", [(1, 1, 1, 1), (2, 2, 2, 2), (4, 1, 0, 1)])
@pytest.mark.parametrize("channels_last", [False, True])
def test_grouped_weight_gradient_matches_autograd(groups, stride, padding, dilation, channels_last):
    from up_platform.backends.metax.determinism import convolution_weight_gradient
    generator = torch.Generator().manual_seed(131)
    x = torch.randn(2, 4, 11, 13, generator=generator)
    w = torch.randn(8, 4 // groups, 3, 3, generator=generator, requires_grad=True)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    y = F.conv2d(x, w, stride=stride, padding=padding, dilation=dilation, groups=groups)
    dy = torch.randn(y.shape, generator=generator)
    expected, = torch.autograd.grad(y, w, dy)
    actual = convolution_weight_gradient(dy, x, w.detach(), [stride] * 2, [padding] * 2, [dilation] * 2, groups)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert actual.dtype == torch.float32


def test_explicit_profile_is_vendor_local_preserves_runtime_and_resets():
    from up_platform import BackendConfigurationError, configure_platform
    from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode
    previous = (torch.backends.cudnn.enabled, torch.backends.mha.get_fastpath_enabled())
    configure_platform({"backend": "cpu"}, purpose="training", reset=True)
    native_owner = configure_platform({"backend": "metax"}, purpose="training", reset=True).capabilities()["operations"]["naive_nms"]["owner"]
    config = {"backend": "metax", "numerics": "metax_deterministic_v1"}
    try:
        provider = configure_platform(config, purpose="training", reset=True)
        assert provider.runtime._numerics is None
        assert provider.numerics_state()["profile"] == "metax_deterministic_v1"
        assert provider.numerics_state()["rng"] == "vendor-native"
        assert provider.capabilities()["operations"]["naive_nms"]["owner"] == native_owner
        assert provider is configure_platform(config, purpose="training")
        with pytest.raises(BackendConfigurationError, match="Reset"):
            configure_platform({"backend": "metax", "numerics": "native"}, purpose="training")

        class Nested(TorchDispatchMode):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                return function(*args, **(kwargs or {}))

        with Nested():
            with pytest.raises(BackendConfigurationError, match="nested"):
                configure_platform({"backend": "cpu"}, purpose="training", reset=True)
    finally:
        configure_platform({"backend": "cpu"}, purpose="training", reset=True)
    assert _get_current_dispatch_mode() is None
    assert previous == (torch.backends.cudnn.enabled, torch.backends.mha.get_fastpath_enabled())


def test_nvidia_native_is_not_changed_by_metax_profile():
    from up_platform import configure_platform, BackendConfigurationError
    try:
        with pytest.raises(BackendConfigurationError, match="Unknown numerical profile"):
            configure_platform({"backend": "nvidia", "numerics": "metax_deterministic_v1"}, purpose="training", reset=True)
        provider = configure_platform({"backend": "nvidia"}, purpose="training", reset=True)
        assert provider.numerics_state()["profile"] == "native"
    finally:
        configure_platform({"backend": "cpu"}, purpose="training", reset=True)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    collector = importlib.import_module("collect_metax_determinism")
    monkeypatch.setattr(collector, "verify_data", lambda value: {"checked": True})
    root, before, after = [tmp_path / name for name in ("results", "before", "after")]
    for source in (before, after):
        (source / "up/models").mkdir(parents=True)
        (source / "up/models/model.py").write_text("pass\n")

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    for name in ("data", "fixed-v1/data"):
        save(root / name / "data.json", {})
    for model in collector.MODELS:
        for label, source in (("baseline", before), ("changed-0", after), ("changed-1", after)):
            for phase in ("training", "inference"):
                save(root / "fixed-v1" / model / label / phase / "report.json", {
                    "passed": True, "source": collector.source_manifest(source), "adaptation": {},
                    "steps": [{"loss": 1., "losses": {"loss": 1.}}], "evaluations": [{"metrics": {"AP": 0.}}]})
        for label in ("changed-0", "changed-1"):
            save(root / "fixed-v1" / model / ("compare-" + label) / "report.json", {
                "passed": True, "rtol": 1e-4, "atol": 1e-5})
        for index in range(4):
            save(root / "candidate-replay" / model / str(index) / "report.json", {
                "rtol": 1e-4, "atol": 1e-5, "providers": {"metax_deterministic_v1": {
                    "exact": True, "repeats": 10, "vs_dnn_on": [{"outside_tolerance": 0, "max_abs": 1e-7}]}}})
    return collector, root, before, after


def test_evidence_rejects_changed_model_and_changed_tolerance(evidence):
    collector, root, before, after = evidence
    assert collector.audit(root, before, after)["passed"]
    path = root / "fixed-v1/resnet18/compare-changed-0/report.json"
    path.write_text(json.dumps({"passed": True, "rtol": 1e-3, "atol": 1e-5}))
    with pytest.raises(AssertionError):
        collector.audit(root, before, after)
    (after / "up/models/model.py").write_text("raise RuntimeError\n")
    with pytest.raises(AssertionError, match="Model source changed"):
        collector.audit(root, before, after)


def test_failed_comparison_cannot_be_published_as_passed(evidence):
    collector, root, before, after = evidence
    path = root / "fixed-v1/resnet18/compare-changed-0/report.json"
    path.write_text(json.dumps({"passed": False, "rtol": 1e-4, "atol": 1e-5}))
    assert not collector.audit(root, before, after)["passed"]


def test_evidence_keeps_failed_controls_hashes_binaries_and_refuses_overwrite(evidence):
    import tarfile
    collector, root, before, after = evidence
    failure = root / "dnn-controls/detr/0/failure.json"
    failure.parent.mkdir(parents=True)
    failure.write_text('{"error": "expected failed negative control"}')
    binary = failure.with_name("gradients-1.pt")
    binary.write_bytes(b"retained on task filesystem")
    archive = root / "evidence.tar.gz"
    assert collector.collect(root, archive, before, after)["passed"]
    index = json.loads(Path(str(archive) + ".index.json").read_text())
    assert index["retained_binary_files"][binary.relative_to(root).as_posix()]["sha256"] == collector.file_hash(binary)
    with tarfile.open(archive) as packed:
        assert failure.relative_to(root).as_posix() in packed.getnames()
        assert binary.relative_to(root).as_posix() not in packed.getnames()
    with pytest.raises(FileExistsError):
        collector.collect(root, archive, before, after)
