"""Compatibility alias for the platform implementation."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("up_platform.base.cuda")
