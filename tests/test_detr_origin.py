"""Attribution diagnostics must not mistake replay drift for a vendor result."""
import importlib
from pathlib import Path
import subprocess
import sys

import pytest
import torch


@pytest.fixture
def tool(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmark/hardware"))
    return importlib.import_module("diagnose_detr_origin")


def test_pure_guard_blocks_both_packages_not_official_model(tool):
    guard = tool.BlockUP(platform=True)
    for name in ("up", "up.runner", "up_platform", "up_platform.backends.metax"):
        with pytest.raises(ImportError, match="forbids importing"):
            guard.find_spec(name)
    assert guard.find_spec("models.detr") is None
    assert guard.find_spec("torch") is None


def test_local_control_may_import_platform_but_not_up(tool):
    guard = tool.BlockUP()
    assert guard.find_spec("up_platform") is None
    with pytest.raises(ImportError):
        guard.find_spec("up.models")


def test_tolerance_and_bitwise_are_distinct(tool):
    assert (tool.RTOL, tool.ATOL) == (1e-4, 1e-5)
    result = tool.tensor_comparison(torch.ones(3), torch.ones(3) + 1e-6)
    assert not result["exact"] and result["outside_tolerance"] == 0
    assert tool.tensor_comparison(torch.zeros(3), torch.ones(3))["outside_tolerance"] == 3


@pytest.mark.parametrize("invalid", [float("inf"), float("nan")])
def test_nonfinite_values_fail(tool, invalid):
    with pytest.raises(ValueError, match="Nonfinite"):
        tool.tensor_comparison(torch.zeros(1), torch.tensor([invalid]))


def test_gradient_none_and_parameter_identity_must_match(tool):
    assert tool.tensor_comparison(None, None)["exact"]
    assert tool.tensor_comparison(None, torch.zeros(1))["outside_tolerance"] == 1
    with pytest.raises(ValueError, match="parameter gradient keys"):
        tool.compare_gradients({"a": torch.ones(1)}, {"b": torch.ones(1)})


def test_digest_keeps_shape_dtype_and_dictionary_content(tool):
    value = {"w": torch.arange(6.).view(2, 3), "optional": None}
    assert tool.digest(value) == tool.digest(tool.cpu(value))
    assert tool.digest(value) != tool.digest({"w": value["w"].flatten(), "optional": None})
    assert tool.digest(torch.ones(2)) != tool.digest(torch.ones(2, dtype=torch.float64))


def test_output_must_not_overwrite_evidence(tool, tmp_path):
    path = tmp_path / "run"
    tool.new_output(path)
    (path / "preserved.txt").write_text("evidence")
    with pytest.raises(FileExistsError):
        tool.new_output(path)
    assert (path / "preserved.txt").read_text() == "evidence"


def test_nested_cpu_mapping_retains_optional_outputs(tool):
    value = ([torch.ones(2), None], {"enabled": False, "x": torch.ones(1)})
    mapped = tool.cpu(value)
    assert isinstance(mapped, tuple) and isinstance(mapped[0], list)
    assert mapped[0][1] is None and mapped[1]["enabled"] is False
    assert mapped[0][0].data_ptr() != value[0][0].data_ptr()


def test_cpu_replay_retains_tensor_layout(tool):
    value = torch.arange(12.).view(3, 4).t()
    restored = tool.restore_layout(value, tool.tensor_layout(value))
    assert restored is value and restored.stride() == (1, 4)


def test_help_does_not_initialize_torch_or_a_device(tool):
    result = subprocess.run([sys.executable, "-I", tool.__file__, "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "trace_step" in result.stdout and not result.stderr


@pytest.mark.parametrize("options,message", [(["--steps", "0"], "720-update"), (["--repeats", "1"], "at least two")])
def test_invalid_budgets_fail_before_any_gpu_work(tool, tmp_path, options, message):
    output = tmp_path / "never-created"
    result = subprocess.run([sys.executable, "-I", tool.__file__, "replay", "--output", str(output), *options],
                            capture_output=True, text=True)
    assert result.returncode == 2 and message in result.stderr
    assert not output.exists()
