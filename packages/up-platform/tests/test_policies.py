"""Numerical experiments are optional policies, not dependencies of native runtime."""
import subprocess
import sys

import pytest
import torch

from up_platform import configure_platform
from up_platform.common.policy import NativePolicy
from up_platform.registry import NUMERICAL_POLICIES


@pytest.fixture(autouse=True)
def reset():
    configure_platform({"backend": "cpu"}, purpose="training", reset=True)
    yield
    configure_platform({"backend": "cpu"}, purpose="training", reset=True)


@pytest.mark.parametrize("backend", ["cpu", "nvidia", "metax"])
@pytest.mark.parametrize("purpose", ["training", "inference"])
def test_native_never_imports_experiments(backend, purpose, tmp_path):
    code = """
import sys
class Guard:
    def find_spec(self, name, *args):
        if name.startswith('up_platform.experimental'):
            raise AssertionError('native imported experiment: ' + name)
sys.meta_path.insert(0, Guard())
import up_platform
ctx = up_platform.configure_platform({'backend': %r}, purpose=%r)
assert ctx.numerics_state()['profile'] == 'native'
assert all(owner != 'portable_v1' for owner, name in ctx._operations)
assert ctx.runtime._numerics is None
""" % (backend, purpose)
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)


def test_policy_is_scoped_to_its_context_and_leaves_registry_native(monkeypatch):
    events = []

    class ExamplePolicy(NativePolicy):
        profile = "test_policy"
        provider = "test_policy"
        operations = {("test_policy", "marker"): {
            "implementation": lambda: "policy", "scope": "common", "backward": False,
            "dtypes": None, "variants": None}}

        def start(self):
            events.append("start")

        def close(self):
            events.append("close")

    monkeypatch.setitem(NUMERICAL_POLICIES, "test_policy", ExamplePolicy)
    context = configure_platform({"backend": "cpu", "numerics": "test_policy"}, purpose="training", reset=True)
    assert context.call("marker") == "policy"
    native = configure_platform({"backend": "cpu"}, purpose="training", reset=True)
    assert events == ["start", "close"]
    assert "marker" not in native.capabilities()["operations"]


def test_failed_activation_restores_modes_and_global_controls(monkeypatch):
    from torch.overrides import _get_current_function_mode
    from torch.utils._python_dispatch import _get_current_dispatch_mode
    from up_platform.experimental.numerics.mode import PortableNumerics
    before = (torch.backends.cudnn.enabled, torch.backends.mha.get_fastpath_enabled(),
              _get_current_function_mode(), _get_current_dispatch_mode())

    def fail(mode):
        raise RuntimeError("activation failed")

    monkeypatch.setattr(PortableNumerics, "__enter__", fail)
    with pytest.raises(RuntimeError, match="activation failed"):
        configure_platform({"backend": "metax", "numerics": "portable_v1"}, purpose="training", reset=True)
    assert before == (torch.backends.cudnn.enabled, torch.backends.mha.get_fastpath_enabled(),
                      _get_current_function_mode(), _get_current_dispatch_mode())
