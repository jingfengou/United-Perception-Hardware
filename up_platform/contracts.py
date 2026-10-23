"""Contract metadata is separate from evidence of execution or validation."""
from dataclasses import dataclass


class PlatformError(RuntimeError):
    pass


class UnsupportedOperation(PlatformError):
    def __init__(self, vendor, purpose, operation, reason):
        super().__init__(f"{vendor}/{purpose}: {operation}: {reason}")


@dataclass(frozen=True)
class Contract:
    source: str
    output: str
    gradients: str
    state: str = "No persistent state is changed."
    device: str = "Inputs and output remain on the selected device; no CPU fallback."
    dtype: str = "Preserve upstream dtype support and PyTorch promotion/autocast rules."
    layout: str = "Preserve upstream shapes, strides and broadcasting requirements."
    empty: str = "Preserve upstream empty-input output or error; do not fabricate output."
    export: str = "Preserve the underlying PyTorch ONNX symbolic and attributes."
    scopes: tuple = ("training", "inference")


def operation(**kwargs):
    contract = Contract(**kwargs)

    def decorate(function):
        function.contract = contract
        return function
    return decorate


@dataclass(frozen=True)
class Capability:
    operation: str
    provider: str
    registered: bool
    dependencies: tuple
    dependency_available: object = None
    executed: bool = False
    verified: bool = False
    reason: str = "No validation evidence is implied by registration."
