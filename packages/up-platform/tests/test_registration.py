"""A new vendor supplies a device, not empty training/inference subclasses."""
import copy
import importlib
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from up_platform import (BackendConfigurationError, BackendSpec, CapabilityError,
                         DeviceBackend, configure_platform, register_backend, register_op)
from up_platform.registry import (BACKENDS, BACKEND_PROBES, OPERATIONS,
                                  INFERENCE_FACTORY, TRAINING_FACTORY)


@pytest.fixture(autouse=True)
def isolate():
    old = dict(BACKENDS), copy.deepcopy(OPERATIONS), dict(BACKEND_PROBES)
    configure_platform({"backend": "cpu"}, purpose="training", reset=True)
    yield
    BACKENDS.clear()
    BACKENDS.update(old[0])
    OPERATIONS.clear()
    OPERATIONS.update(old[1])
    BACKEND_PROBES.clear()
    BACKEND_PROBES.update(old[2])
    configure_platform({"backend": "cpu"}, purpose="training", reset=True)


@pytest.mark.parametrize("purpose", ("training", "inference"))
@pytest.mark.parametrize("form", ("direct", "decorator", "options_decorator"))
def test_device_only_registration_uses_shared_execution(purpose, form):
    class VendorDevice(DeviceBackend):
        name = "simple_vendor"

    if form == "direct":
        assert register_backend(VendorDevice) is VendorDevice
    elif form == "decorator":
        @register_backend
        class VendorDevice(VendorDevice):
            pass
    else:
        @register_backend()
        class VendorDevice(VendorDevice):
            pass
    assert BACKENDS["simple_vendor"].device_factory is VendorDevice
    context = configure_platform({"backend": "simple_vendor"}, purpose=purpose, reset=True)
    expected = TRAINING_FACTORY if purpose == "training" else INFERENCE_FACTORY
    module, name = expected.split(":")
    assert type(context.execution) is getattr(importlib.import_module(module), name)
    boxes = torch.tensor([[0., 0., 4., 4., .9], [0., 0., 4., 4., .8]])
    assert context.ops.naive_nms(boxes, .5).tolist() == [0]
    assert context.require("naive_nms")["owner"] == "portable"


def test_override_only_training_and_reuse_inference():
    from up_platform.training.backend import TrainingBackend

    class CustomTraining(TrainingBackend):
        def autocast(self, enabled=True, **kwargs):
            return "vendor training"

    @register_backend(training=CustomTraining)
    class VendorDevice(DeviceBackend):
        name = "training_override"

    context = configure_platform({"backend": VendorDevice.name}, purpose="training", reset=True)
    assert context.autocast() == "vendor training"
    context = configure_platform({"backend": VendorDevice.name}, purpose="inference", reset=True)
    assert type(context.execution).__module__ == "up_platform.inference.backend"
    with context.autocast(enabled=False):
        pass


@pytest.mark.parametrize("purpose,other", (("training", "inference"), ("inference", "training")))
def test_optional_execution_factory_is_lazy(purpose, other):
    @register_backend(**{other: "missing_vendor_execution_sdk:Backend"})
    class VendorDevice(DeviceBackend):
        name = "lazy_vendor"

    context = configure_platform({"backend": VendorDevice.name}, purpose=purpose, reset=True)
    assert context.purpose == purpose
    with pytest.raises(BackendConfigurationError, match="missing_vendor_execution_sdk"):
        configure_platform({"backend": VendorDevice.name}, purpose=other, reset=True)


@pytest.mark.parametrize("disabled", ("training", "inference"))
def test_explicit_none_disables_only_the_requested_purpose(disabled):
    @register_backend(**{disabled: None})
    class VendorDevice(DeviceBackend):
        name = "single_purpose"

    with pytest.raises(CapabilityError, match="no execution backend"):
        configure_platform({"backend": VendorDevice.name}, purpose=disabled, reset=True)
    other = "inference" if disabled == "training" else "training"
    assert configure_platform({"backend": VendorDevice.name}, purpose=other, reset=True).purpose == other


def test_legacy_spec_none_and_return_value_are_unchanged():
    spec = BackendSpec("legacy_device_only", "cpu", DeviceBackend)
    assert register_backend(spec) is spec
    for purpose in ("training", "inference"):
        with pytest.raises(CapabilityError, match="no execution backend"):
            configure_platform({"backend": spec.name}, purpose=purpose, reset=True)
    with pytest.raises(TypeError, match="BackendSpec itself"):
        register_backend(spec, training=TRAINING_FACTORY, replace=True)


@pytest.mark.parametrize("invalid", (object(), type("MissingMetadata", (), {})))
def test_registration_rejects_invalid_device_metadata(invalid):
    with pytest.raises(TypeError, match="device backend class"):
        register_backend(invalid)


def test_class_metadata_and_vendor_requirements_are_preserved():
    class VendorDevice(DeviceBackend):
        name = "restricted_vendor"
        op_fallbacks = ("portable", "vendor_shared")
        requirements = {"python": ">=99"}

    register_backend(VendorDevice)
    spec = BACKENDS[VendorDevice.name]
    assert spec.shared_providers == VendorDevice.op_fallbacks
    assert spec.requirements == VendorDevice.requirements
    assert spec.requirements is not VendorDevice.requirements
    with pytest.raises(BackendConfigurationError, match="requires python>=99"):
        configure_platform({"backend": VendorDevice.name, "requirements": {"python": ">=3"}},
                           purpose="training", reset=True)


def test_single_op_override_in_a_distinct_device_namespace():
    @register_backend
    class VendorDevice(DeviceBackend):
        name = "different_namespace"
        device_type = "privateuseone"

    register_op(VendorDevice.name, "gpu_iou_overlap", lambda a, b: (a, b), scope="common")
    for purpose in ("training", "inference"):
        context = configure_platform({"backend": VendorDevice.name, "device": "privateuseone:1"},
                                     purpose=purpose, reset=True)
        assert str(context.device) == "privateuseone:1"
        assert context.ops.gpu_iou_overlap(1, 2) == (1, 2)
        assert context.require("naive_nms")["owner"] == "portable"
        with pytest.raises(CapabilityError, match="roi_align"):
            context.require("roi_align")
    # No privateuseone computation is claimed: this test checks routing only.


def test_class_registration_keeps_duplicate_replace_and_probe_contracts():
    class VendorDevice(DeviceBackend):
        name = "probed_vendor"

    probe = lambda: True
    register_backend(VendorDevice, probe=probe)
    with pytest.raises(ValueError, match="already registered"):
        register_backend(VendorDevice)
    assert BACKEND_PROBES[VendorDevice.name] is probe
    assert register_backend(VendorDevice, replace=True) is VendorDevice
    assert VendorDevice.name not in BACKEND_PROBES


def test_old_factory_module_paths_are_compatibility_aliases():
    from up_platform.inference.backend import InferenceBackend
    from up_platform.training.backend import TrainingBackend

    assert importlib.import_module("up_platform.backends.cpu.device").CpuDevice is DeviceBackend
    for vendor in ("cpu", "nvidia", "metax", "ascend", "cambricon"):
        assert importlib.import_module("up_platform.backends." + vendor + ".training").Backend is TrainingBackend
        assert importlib.import_module("up_platform.backends." + vendor + ".inference").Backend is InferenceBackend
        assert BACKENDS[vendor].training_factory == TRAINING_FACTORY
        assert BACKENDS[vendor].inference_factory == INFERENCE_FACTORY


@pytest.mark.parametrize("vendor", ("cpu", "nvidia", "metax"))
@pytest.mark.parametrize("purpose,other", (("training", "inference"), ("inference", "training")))
def test_builtin_initialization_bypasses_compatibility_shells(vendor, purpose, other, tmp_path):
    code = """
import sys
import up_platform
class Guard:
    def find_spec(self, fullname, *args):
        if fullname == 'up' or fullname.startswith('up.'):
            raise AssertionError('unexpected UP import: ' + fullname)
        if fullname.startswith('up_platform.backends.') and fullname.rsplit('.', 1)[-1] in ('training', 'inference'):
            raise AssertionError('unexpected compatibility module: ' + fullname)
        if fullname.startswith('up_platform.{other}'):
            raise AssertionError('unexpected opposite purpose: ' + fullname)
sys.meta_path.insert(0, Guard())
ctx = up_platform.configure_platform({{'backend': '{vendor}'}}, purpose='{purpose}')
assert ctx.execution.__class__.__module__ == 'up_platform.{purpose}.backend'
assert ctx.purpose == '{purpose}'
""".format(vendor=vendor, purpose=purpose, other=other)
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)


@pytest.mark.parametrize("purpose", ("training", "inference"))
def test_three_file_vendor_example(purpose, tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples/vendor_plugin"
    code = """
import sys
sys.path.insert(0, {example!r})
import torch
from up_platform import configure_platform
ctx = configure_platform({{'backend': 'example_vendor', 'plugins': ['up_example_vendor']}}, purpose='{purpose}')
boxes = torch.tensor([[0., 0., 4., 4., .9], [0., 0., 4., 4., .8]])
assert ctx.ops.naive_nms(boxes, .5).tolist() == [0]
assert ctx.require('naive_nms')['owner'] == 'example_vendor'
assert ctx.require('gpu_iou_overlap')['owner'] == 'portable'
assert 'up_example_vendor.training' not in sys.modules
assert 'up_example_vendor.inference' not in sys.modules
assert 'up' not in sys.modules
""".format(example=str(example), purpose=purpose)
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)
