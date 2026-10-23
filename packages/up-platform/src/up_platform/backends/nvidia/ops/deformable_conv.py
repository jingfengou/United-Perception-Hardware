"""Compatibility alias for the operator-category implementation."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("up_platform.ops.deform_conv.nvidia.deformable_conv")
