"""Executable CPU demonstration. Replace Device/SDK with a real vendor runtime."""
from up_platform import BackendSpec, register_backend, register_op

from .device import ExampleDevice
from .ops import naive_nms

register_backend(BackendSpec(
    name="example_vendor", device_type="cpu", device_factory=ExampleDevice,
    training_factory="up_example_vendor.training:Backend",
    inference_factory="up_example_vendor.inference:Backend",
    shared_providers=("portable",)))
register_op("example_vendor", "naive_nms", naive_nms, scope="common", backward=False,
            conventions=("offset",), variants={"offset": (0, 1)}, dependencies=("torch",))
