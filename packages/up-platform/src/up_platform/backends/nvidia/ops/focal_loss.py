"""Compatibility alias for the operator-category implementation."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("up_platform.ops.focal_loss.nvidia.focal_loss")
