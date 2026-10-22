"""Compatibility alias; implementation lives in the independent platform package."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("up_platform.experimental.numerics.mode")
