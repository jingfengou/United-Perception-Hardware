"""Compatibility alias; implementation lives in the independent platform package."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("up_platform.backends.nvidia.ops.roiaware_pool3d_utils")
