"""The category-first architecture works without legacy implementation modules."""
import importlib
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from up_platform import CapabilityError, configure_platform
from up_platform.catalog import NVIDIA_MODULES


@pytest.mark.parametrize("name", sorted(set(NVIDIA_MODULES) - {"rotate_iou"}))
def test_old_nvidia_module_is_the_same_category_implementation(name):
    current = importlib.import_module("up_platform.ops." + NVIDIA_MODULES[name])
    old = importlib.import_module("up_platform.backends.nvidia.ops." + name)
    assert old is current


@pytest.mark.parametrize("old,current", [
    ("registry", "registry_factory"),
    ("common.runtime", "base.base"),
    ("shared.layers", "ops.normalization.common"),
    ("shared.sparse", "ops.sparse.common"),
    ("training.backend", "training.runtime.backend.base"),
    ("backends.metax.determinism", "ops.convolution.metax.weight_gradient"),
    ("backends.metax.determinism_v2", "ops.mm.metax.determinism"),
    ("backends.metax.fixed_gemm", "ops.mm.metax.fixed_gemm"),
])
def test_old_addresses_do_not_create_duplicate_classes_or_state(old, current):
    assert importlib.import_module("up_platform." + old) is importlib.import_module("up_platform." + current)


@pytest.mark.parametrize("purpose", ["training", "inference"])
def test_selected_purpose_factory_and_public_operation_facade(purpose):
    from up_platform import ops
    context = configure_platform({"backend": "cpu"}, purpose=purpose, reset=True)
    factory = importlib.import_module("up_platform.training.runtime.backend.factory" if purpose == "training"
                                      else "up_platform.inference.factory")
    assert factory.get_backend() is context.execution
    boxes = torch.tensor([[0., 0., 2., 2., .9], [0., 0., 2., 2., .8]])
    assert ops.naive_nms(boxes, .5).tolist() == [0]
    assert context.require("naive_nms")["implementation"] == "up_platform.ops.nms.common:naive_nms"
    other = importlib.import_module("up_platform.inference.factory" if purpose == "training"
                                    else "up_platform.training.runtime.backend.factory")
    with pytest.raises(CapabilityError):
        other.get_backend()


def test_all_native_catalog_modules_are_category_local():
    import up_platform
    root = Path(up_platform.__file__).parent
    for relative in NVIDIA_MODULES.values():
        assert relative.split(".")[1] == "nvidia"
        assert (root / "ops" / (relative.replace(".", "/") + ".py")).is_file()


@pytest.mark.parametrize("purpose,opposite", [("training", "inference"), ("inference", "training")])
def test_runtime_does_not_load_legacy_or_opposite_purpose_modules(purpose, opposite, tmp_path):
    code = """
import sys
import up_platform
class Guard:
    def find_spec(self, fullname, *args):
        if (fullname.startswith('up_platform.backends.') or fullname.startswith('up_platform.{opposite}')
                or fullname in ('up_platform.common.runtime', 'up_platform.shared.execution',
                                'up_platform.training.backend', 'up_platform.shared.portable')):
            raise AssertionError('unexpected implementation import: ' + fullname)
sys.meta_path.insert(0, Guard())
ctx = up_platform.configure_platform({{'backend': 'metax'}}, purpose='{purpose}')
assert ctx.name == 'metax' and ctx.device_type == 'cuda'
assert not any(name.startswith('up_platform.experimental.') for name in sys.modules)
assert 'triton' not in sys.modules
""".format(purpose=purpose, opposite=opposite)
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)
