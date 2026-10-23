"""UP hardware contracts. Importing this module does not import PyTorch or a SDK."""
from .factory import configure_platform, get_platform, register_backend

__all__ = ["configure_platform", "get_platform", "register_backend"]
