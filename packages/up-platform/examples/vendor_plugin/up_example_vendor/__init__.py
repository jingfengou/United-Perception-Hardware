"""Executable CPU demonstration. Replace Device/SDK with a real vendor runtime."""
from up_platform import register_backend, register_op

from .device import ExampleDevice
from .ops import naive_nms

register_backend(ExampleDevice)
register_op("example_vendor", "naive_nms", naive_nms, scope="common", backward=False,
            conventions=("offset",), variants={"offset": (0, 1)}, dependencies=("torch",))
