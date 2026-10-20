class CapabilityError(RuntimeError):
    """A selected backend cannot implement a requested hardware operation."""


class BackendConfigurationError(RuntimeError):
    """Hardware identity, runtime configuration, or environment is inconsistent."""
